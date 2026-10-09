"""Validate current Kaggle -> Drive eval layout without a real model, API or Drive mount.

Run: python3 datasets/code/test_eval_colab_kaggle_inputs_offline.py
"""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import test_soft_scoring_offline as fixtures


NB=fixtures.FILES[2]
RUN_ID='20261008_033225_3733263d'


def input_namespace(directory):
    ns=fixtures.common_namespace();nb=fixtures.notebook(NB)
    fixtures.definitions(nb,3,ns)
    data=[ns['load_rows'](fixtures.ROOT/p) for p in fixtures.DATA]
    fit,val,seen,unseen,manifest=ns['prepare_data_splits'](*data)
    root=Path(directory);results=root/'results';results.mkdir()
    adapter=root/'training_runs'/RUN_ID/'best_adapter';adapter.mkdir(parents=True)
    metadata=adapter.parent/'results';metadata.mkdir()
    (metadata/'best_validation.json').write_text(json.dumps({'step':93,'fcr':.25,'hsr':.5}))
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'Qwen/Qwen3.5-0.8B'}))
    (adapter/'adapter_model.safetensors').write_text('fake weights; never loaded')
    ns.update(RUN_DIR=str(root),RESULTS_DIR=str(results),ADAPTER_DIR=str(adapter),ADAPTER_VALIDATION_FILE='',
              BASELINE_IMPORT_FILE=str(results/'baseline_outputs.json'),EVALUATE_BENCHMARK_BASELINE=True,
              TRAIN_ROWS=fit,VALIDATION_ROWS=val[:32],TEST_SEEN_ROWS=seen,TEST_UNSEEN_ROWS=unseen,
              SPLIT_MANIFEST=manifest,SPLIT_SIGNATURE='fixture-split',MODEL_DEVICE='cpu',
              tokenizer=SimpleNamespace(pad_token_id=1,bos_token_id=None,eos_token_id=2),
              torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),bfloat16='bf16',float32='fp32'),
              set_seed=lambda seed:None)
    # All cached base responses; no generation should happen when loading/evaluating inputs.
    fixtures.definitions(nb,4,ns)
    cache={ns['baseline_key'](split,row):{'text':'cached base output','generated_tokens':4,'truncated':False}
           for split,rows in [('seen',seen),('unseen',unseen)] for row in rows}
    ns['save_json'](ns['BASELINE_IMPORT_FILE'],cache)
    loads=[];adapted=[]
    model=SimpleNamespace(config=SimpleNamespace(get_text_config=lambda:SimpleNamespace(),use_cache=False),
                          generation_config=SimpleNamespace(eos_token_id=2),
                          model=SimpleNamespace(register_forward_pre_hook=lambda *a,**kw:None),
                          eval=lambda:None,gradient_checkpointing_disable=lambda:None)
    def load_base(name,**kw):loads.append((name,kw));return model
    def load_adapter(base,path,**kw):adapted.append((base,path,kw));return model
    ns.update(Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load_base),
              PeftModel=SimpleNamespace(from_pretrained=load_adapter))
    source=ast.parse(''.join(nb['cells'][4]['source']))
    statements=[node for node in source.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    code=compile(ast.Module(body=statements,type_ignores=[]),'full offline title 4','exec')
    return ns,cache,code,loads,adapted


class EvalKaggleInputTests(unittest.TestCase):
    def test_current_defaults(self):
        nb=fixtures.notebook(NB)
        source=''.join(nb['cells'][2]['source'])
        self.assertIn("KAGGLE_TRAIN_RUN_ID = '"+RUN_ID+"'",source)
        self.assertIn("GROQ_MODEL = 'openai/gpt-oss-120b'",source)
        self.assertIn("ADAPTER_DIR = os.path.join(KAGGLE_TRAIN_RUN_DIR, 'best_adapter')",source)
        self.assertIn("BASELINE_IMPORT_FILE = os.path.join(RESULTS_DIR, 'baseline_outputs.json')",source)
        self.assertIn("SPLIT_MANIFEST_IMPORT_FILE = os.path.join(RESULTS_DIR, 'split_manifest.json')",source)
        self.assertIn('GROQ_MAX_RETRY_WAIT_SECONDS = 300.0',source)
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),NB+str(i),'exec')
                self.assertIsNone(cell.get('execution_count'));self.assertFalse(cell.get('outputs'))

    def test_full_adapter_load_reads_step_93_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            ns,cache,code,loads,adapted=input_namespace(directory)
            before=Path(ns['BASELINE_IMPORT_FILE']).read_bytes()
            with redirect_stdout(StringIO()) as output:exec(code,ns)
            self.assertEqual(ns['ADAPTER_STEP'],93)
            self.assertEqual(ns['baseline_outputs'],cache)
            self.assertEqual(before,Path(ns['BASELINE_IMPORT_FILE']).read_bytes())
            self.assertEqual(len(loads),1);self.assertEqual(loads[0][0],'Qwen/Qwen3.5-0.8B')
            self.assertEqual(adapted[0][1],ns['ADAPTER_DIR']);self.assertFalse(adapted[0][2]['is_trainable'])
            self.assertIn('step=93',output.getvalue())
            self.assertIsNone(ns['TRAIN_PROMPT_HASHES'])  # No fictitious training provenance.
            # Record exact imported run, adapter step and chosen judge in eval output metadata.
            ns.update(KAGGLE_TRAIN_RUN_ID=RUN_ID,MAX_PROMPT_LENGTH=2048,GROQ_MODEL='openai/gpt-oss-120b',
                      JUDGE_MODEL='openai/gpt-oss-120b',JUDGE_PROVIDER='groq',JUDGE_BASE_URL='https://api.groq.com/openai/v1',
                      GROQ_BASE_URL='https://api.groq.com/openai/v1',JUDGE_RUBRIC_VERSION='soft-v2',SOFT_REWARD_GATE='all_hard',
                      GROQ_MAX_REQUESTS=4,GROQ_WINDOW_SECONDS=60.,GROQ_MIN_INTERVAL_SECONDS=10.,GROQ_MAX_RETRY_WAIT_SECONDS=300.,
                      JUDGE_MAX_TOKENS=2048)
            prefix=''.join(fixtures.notebook(NB)['cells'][7]['source']).split('def hard_eval')[0]
            exec(prefix,ns)
            config=json.loads((Path(ns['EVAL_RESULTS_DIR'])/'eval_config.json').read_text())
            self.assertEqual(config['adapter_step'],93)
            self.assertEqual(config['kaggle_train_run_id'],RUN_ID)
            self.assertEqual(config['groq_request_config']['model'],'openai/gpt-oss-120b')
            self.assertEqual(config['groq_request_config']['max_retry_wait_seconds'],300.)

    def test_missing_baseline_or_bad_metadata_stops_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            ns,cache,code,loads,adapted=input_namespace(directory)
            key=next(iter(cache));cache.pop(key);ns['save_json'](ns['BASELINE_IMPORT_FILE'],cache)
            with self.assertRaisesRegex(ValueError,'Thiếu baseline'):exec(code,ns)
            self.assertFalse(loads);self.assertFalse(adapted)
        for step in ('93',True,-1,None):
            with self.subTest(step=step),tempfile.TemporaryDirectory() as directory:
                ns,_,code,loads,adapted=input_namespace(directory)
                record=Path(ns['ADAPTER_DIR']).parent/'results/best_validation.json';record.write_text(json.dumps({'step':step}))
                with redirect_stdout(StringIO()),self.assertRaisesRegex(ValueError,'step nguyên'):exec(code,ns)
                self.assertFalse(loads);self.assertFalse(adapted)

    def test_explicit_metadata_path_and_missing_record_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            ns,_,code,_,_=input_namespace(directory)
            record=Path(ns['ADAPTER_DIR']).parent/'results/best_validation.json';record.unlink()
            with redirect_stdout(StringIO()):exec(code,ns)
            self.assertIsNone(ns['ADAPTER_STEP'])
            flat=Path(directory)/'best_validation.json';flat.write_text(json.dumps({'step':93}))
            ns['ADAPTER_VALIDATION_FILE']=str(flat)
            with redirect_stdout(StringIO()):exec(code,ns)
            self.assertEqual(ns['ADAPTER_STEP'],93)


if __name__=='__main__':
    unittest.main(verbosity=2)
