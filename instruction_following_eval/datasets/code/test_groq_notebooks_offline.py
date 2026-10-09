"""Groq notebook integration tests: fake secrets/API, virtual time and temporary outputs only.

Run: python3 datasets/code/test_groq_notebooks_offline.py
"""
import ast
import copy
import csv
import random
from collections import Counter
import importlib.metadata
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import threading
import types
import unittest
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_soft_scoring_offline as fixtures

FILES=fixtures.FILES[1:]


def groq_namespace(name,directory):
    ns=fixtures.judge_namespace(name)
    now=[0.];waits=[]
    def sleep(seconds):waits.append(seconds);now[0]+=seconds
    ns.update(JUDGE_PROVIDER='groq',JUDGE_MODEL=ns['GROQ_MODEL'],JUDGE_BASE_URL=ns['GROQ_BASE_URL'],
              JUDGE_CACHE_FILE=str(Path(directory)/'soft_judge_cache_v2.jsonl'),
              _JUDGE_PROVIDER_CALLS={'gemma':0,'gemini':0,'openrouter':0,'groq':0})
    ns['_GROQ_LIMITER']=ns['GroqRequestLimiter'](4,60,10,clock=lambda:now[0],sleeper=sleep)
    return ns,now,waits


class GroqNotebookTests(unittest.TestCase):
    def test_full_config_and_judge_cell_all_three_platforms(self):
        for name in FILES:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);real_path=pathlib.Path
                def mapped(value):
                    value=str(value)
                    if value.startswith('/kaggle/'):return str(root/value.lstrip('/'))
                    if value.startswith('/content/drive/MyDrive'):return str(root/'drive')+value[len('/content/drive/MyDrive'):]
                    return value
                class VirtualPath(type(real_path())):
                    def __new__(cls,*parts):
                        return super().__new__(cls,*(mapped(x) for x in parts))
                    def __init__(self,*parts):
                        if real_path.__init__ is not object.__init__:
                            super().__init__(*(mapped(x) for x in parts))
                inputs=root/'kaggle/input/fixture';inputs.mkdir(parents=True)
                for filename in ['train_seen.jsonl','test_seen.jsonl','test_unseen.jsonl','verifier_engine_vi.py']:
                    (inputs/filename).write_text('fixture')
                drive_base='/content/drive/MyDrive/vinfast/Tuan4_5_6'
                run_id='20261008_033225_3733263d' if name==fixtures.FILES[2] else 'mock'
                adapter=drive_base+'/qwen_grpo_v2/training_runs/'+run_id+'/best_adapter'
                real_path(mapped(adapter)).mkdir(parents=True)
                real_path(mapped(adapter+'/adapter_config.json')).write_text(json.dumps({'base_model_name_or_path':'Qwen/Qwen3.5-0.8B'}))
                real_path(mapped(adapter+'/adapter_model.safetensors')).write_text('mock')
                results=real_path(mapped(drive_base+'/qwen_grpo_v2/results'));results.mkdir()
                for relative in ('grpo/train_seen.jsonl','eval/test_seen.jsonl','eval/test_unseen.jsonl','verifier_engine_vi.py'):
                    target=real_path(mapped(drive_base+'/'+relative));target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture')
                (results/'baseline_outputs.json').write_text('{}')
                (results/'split_manifest.json').write_text('{}')
                (results/'latest_training_attempt.json').write_text(json.dumps({'status':'complete','attempt_dir':adapter.rsplit('/',1)[0]}))
                fake_os=types.ModuleType('os');fake_os.__dict__.update(os.__dict__);fake_os.environ={}
                fake_os.makedirs=lambda path,**kw:os.makedirs(mapped(path),**kw)
                fake_os.path=SimpleNamespace(**{x:getattr(os.path,x) for x in dir(os.path) if not x.startswith('__')})
                fake_os.path.isdir=lambda path:os.path.isdir(mapped(path))
                ns=fixtures.common_namespace();ns['torch']=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))
                nb=fixtures.notebook(name)
                with patch.dict(sys.modules,{'os':fake_os}),patch.object(pathlib,'Path',VirtualPath),redirect_stdout(StringIO()):
                    exec(''.join(nb['cells'][2]['source']),ns)
                self.assertEqual(ns['JUDGE_PROVIDER'],'groq')
                expected_model='openai/gpt-oss-120b' if name==fixtures.FILES[2] else 'llama-3.3-70b-versatile'
                self.assertEqual(ns['JUDGE_MODEL'],expected_model)
                self.assertEqual(ns['JUDGE_BASE_URL'],'https://api.groq.com/openai/v1')
                self.assertEqual(ns['JUDGE_MAX_WORKERS'],1)
                self.assertEqual(ns['JUDGE_MAX_TOKENS'],2048)
                self.assertEqual(ns['SOFT_REWARD_GATE'],'all_hard')
                # Entire title 6, including initialization and key selection.
                fixtures.definitions(nb,3,ns)
                calls=[];secret_reads=[];created=[];now=[0.]
                secret='gsk_OFFLINE_TEST_KEY'
                def get_secret(label):
                    secret_reads.append(label)
                    if label!='GROQ_API_KEY':raise AssertionError('Unexpected other provider secret')
                    return secret
                module=types.ModuleType('kaggle_secrets')
                module.UserSecretsClient=lambda:SimpleNamespace(get_secret=get_secret)
                google=types.ModuleType('google');colab=types.ModuleType('google.colab')
                colab.userdata=SimpleNamespace(get=get_secret);google.colab=colab
                def create(**kw):
                    calls.append(kw)
                    return fixtures.fake_response(json.loads(kw['messages'][1]['content'])['criteria'])
                def factory(**kw):
                    created.append(kw);result=fixtures.client(create);result.api_key=secret;return result
                ns.update(os=fake_os,Path=real_path,math=math,re=re,OpenAI=factory,
                          APIStatusError=fixtures.StatusError,APIConnectionError=fixtures.ConnectionError,
                          APITimeoutError=fixtures.TimeoutError,threading=threading,deque=deque,Future=Future,
                          ThreadPoolExecutor=ThreadPoolExecutor,
                          JUDGE_CACHE_FILE=str(root/'cache.jsonl'),
                          time=SimpleNamespace(monotonic=lambda:now[0],sleep=lambda sec:now.__setitem__(0,now[0]+sec)))
                tree=ast.parse(''.join(nb['cells'][6]['source']))
                statements=[x for x in tree.body if not isinstance(x,(ast.Import,ast.ImportFrom))]
                with patch.dict(sys.modules,{'kaggle_secrets':module,'google':google,'google.colab':colab}),redirect_stdout(StringIO()):
                    exec(compile(ast.Module(body=statements,type_ignores=[]),name,'exec'),ns)
                    self.assertEqual(secret_reads,['GROQ_API_KEY']);self.assertFalse(calls)
                    self.assertEqual(created[0]['base_url'],'https://api.groq.com/openai/v1')
                    self.assertEqual(created[0]['max_retries'],0)
                    criterion=[fixtures.criterion()]
                    result=ns['evaluate_single_soft_details']('answer','task',criterion)
                    self.assertEqual(result['provider'],'groq');self.assertEqual(result['score'],1.)
                    request=calls[0]
                    self.assertEqual(request['model'],expected_model)
                    self.assertEqual(request['response_format'],{'type':'json_object'})
                    self.assertEqual(request['max_completion_tokens'],2048)
                    for field in ['extra_body','reasoning','reasoning_effort','tools','max_tokens']:
                        self.assertNotIn(field,request)
                    key=ns['_compute_cache_key']('task','answer',criterion)
                    self.assertNotEqual(key,ns['_compute_cache_key']('task','answer',criterion,'gemma'))
                    ns['GROQ_MODEL']='other-model'
                    self.assertNotEqual(key,ns['_compute_cache_key']('task','answer',criterion));ns['GROQ_MODEL']=expected_model
                    limiter=ns['_GROQ_LIMITER'];sdk=ns['_GROQ_CLIENT']
                    exec(compile(ast.Module(body=statements,type_ignores=[]),name,'exec'),ns)
                    self.assertIs(limiter,ns['_GROQ_LIMITER']);self.assertIs(sdk,ns['_GROQ_CLIENT'])
                    self.assertEqual(ns['evaluate_single_soft_details']('answer','task',criterion)['score'],1.)
                    self.assertEqual(len(calls),1);self.assertEqual(secret_reads,['GROQ_API_KEY'])
                    ns['TRAIN_ROWS']=[{'prompt':'task','soft':json.dumps(criterion)}]
                    if name!=fixtures.FILES[2]:
                        ns['run_soft_judge_preflight']();ns['run_soft_judge_preflight']()
                        self.assertEqual(len(calls),2)  # One new preflight, then cached.
                    else:
                        self.assertNotIn('run_soft_judge_preflight',ns)
                    self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['gemini'],0)
                    self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['gemma'],0)
                    # Changing a live limiter must not discard the old request history.
                    ns['GROQ_MAX_REQUESTS']=5
                    with self.assertRaisesRegex(ValueError,'restart runtime'):
                        exec(compile(ast.Module(body=statements,type_ignores=[]),name,'exec'),ns)
                if 'eval_' in name:
                    calibration=''.join(nb['cells'][9]['source'])
                    self.assertIn('CALIBRATION_PROVIDERS = [JUDGE_PROVIDER]',calibration)

    def test_retry_cooldown_headers_errors_and_no_fake_rewards(self):
        for name in FILES:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                ns,now,waits=groq_namespace(name,directory);criteria=[fixtures.criterion()];calls=[];starts=[]
                key='gsk_OFFLINE_SECRET'
                def create(**kw):
                    calls.append(kw);starts.append(now[0])
                    if len(calls)==1:raise fixtures.StatusError(429,23)
                    return fixtures.fake_response(criteria)
                ns['_GROQ_CLIENT']=fixtures.client(create);ns['_GROQ_CLIENT'].api_key=key
                ns['_GROQ_CLIENT_CONFIG']=(ns['GROQ_BASE_URL'],60,'GROQ_API_KEY')
                with redirect_stdout(StringIO()):
                    for i in range(8):self.assertEqual(ns['_request_groq_score'](str(i),'task',criteria)['score'],1.)
                self.assertGreaterEqual(starts[1]-starts[0],23)
                self.assertTrue(all(b-a>=10 for a,b in zip(starts,starts[1:])))
                for instant in starts:self.assertLessEqual(sum(instant-60<t<=instant for t in starts),4)
                e=fixtures.StatusError(429);e.response.headers={'x-ratelimit-remaining-tokens':'0','x-ratelimit-reset-tokens':'1m2.5s'}
                self.assertEqual(ns['_groq_retry_delay'](e,1,429),62.5)
                e.response.headers={'Retry-After':'80'};self.assertEqual(ns['_groq_retry_delay'](e,1,429),80)
                for code in (400,401,403,404):
                    e=fixtures.StatusError(code);e.args=(f'Invalid {key} gsk_OTHER_KEY',)
                    ns['_GROQ_CLIENT'].chat.completions.create=lambda **kw:(_ for _ in ()).throw(e)
                    before=ns['_JUDGE_PROVIDER_CALLS']['groq']
                    with self.assertRaisesRegex(RuntimeError,f'HTTP {code}') as caught:
                        ns['evaluate_single_soft_details']('answer-'+str(code),'task',criteria)
                    self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['groq'],before+1)
                    self.assertNotIn(key,str(caught.exception));self.assertNotIn('gsk_OTHER_KEY',str(caught.exception))
                    self.assertFalse(ns['_SOFT_JUDGE_CACHE']);self.assertFalse(ns['_JUDGE_INFLIGHT'])
                # A generated invalid-JSON HTTP 400 may be retried, an invalid parameter 400 may not.
                e=fixtures.StatusError(400);e.body={'error':{'code':'json_validate_failed'}};bad=[e]
                def eventually_valid(**kw):
                    if bad:raise bad.pop()
                    return fixtures.fake_response(criteria)
                ns['_GROQ_CLIENT'].chat.completions.create=eventually_valid
                with redirect_stdout(StringIO()):self.assertEqual(ns['_request_groq_score']('x','task',criteria)['score'],1.)
                # Refuse hours of blocking for exhausted daily/token quota.
                e=fixtures.StatusError(429);e.response.headers={'x-ratelimit-remaining-requests':'0','x-ratelimit-reset-requests':'2h'}
                ns['_GROQ_CLIENT'].chat.completions.create=lambda **kw:(_ for _ in ()).throw(e)
                before=ns['_JUDGE_PROVIDER_CALLS']['groq']
                with self.assertRaisesRegex(RuntimeError,'vượt giới hạn retry'):ns['_request_groq_score']('x','task',criteria)
                self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['groq'],before+1)
                # Every invalid schema is rejected; preserve original cause + redacted diagnostic file.
                invalid=[fixtures.fake_response(criteria),fixtures.fake_response(criteria),fixtures.fake_response(criteria)]
                invalid[0].choices[0].finish_reason='length'
                invalid[1].choices[0].message.content='plain text '+key
                obj=json.loads(invalid[2].choices[0].message.content);obj['scores'][0]['reason']='';invalid[2].choices[0].message.content=json.dumps(obj)
                for index,response in enumerate(invalid):
                    ns['_GROQ_CLIENT'].chat.completions.create=lambda **kw:response
                    before=ns['_JUDGE_PROVIDER_CALLS']['groq']
                    with redirect_stdout(StringIO()),self.assertRaises(RuntimeError) as caught:
                        ns['evaluate_single_soft_details']('invalid-'+str(index),'task',criteria)
                    self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['groq'],before+3)
                    self.assertIn('Log:',str(caught.exception))
                    self.assertFalse(ns['_SOFT_JUDGE_CACHE']);self.assertFalse(ns['_JUDGE_INFLIGHT'])
                    self.assertFalse(Path(ns['JUDGE_CACHE_FILE']).exists())
                log=(Path(directory)/'groq_judge_failures_v2.jsonl').read_text()
                self.assertNotIn(key,log);self.assertNotIn('gsk_OTHER_KEY',log)
                self.assertIn('finish_reason=\'length\'',log);self.assertIn('1–400',log)

    def test_groq_eval_outputs_and_calibration(self):
        import numpy as np
        import pandas as pd
        for name,index in [(fixtures.FILES[2],7),(fixtures.FILES[3],10)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                ns,_,_=groq_namespace(name,directory);nb=fixtures.notebook(name)
                fixtures.definitions(nb,index,ns)
                calls=[]
                def create(**kw):
                    calls.append(kw)
                    return fixtures.fake_response(json.loads(kw['messages'][1]['content'])['criteria'])
                ns['_GROQ_CLIENT']=fixtures.client(create)
                ns['_GROQ_CLIENT_CONFIG']=(ns['GROQ_BASE_URL'],60,'GROQ_API_KEY')
                values=[json.loads(l) for l in (fixtures.ROOT/fixtures.DATA[1]).read_text().splitlines()[:2]]
                rows=[ns['normalize_row'](r) for r in values]
                baseline={ns['baseline_key']('seen',r):{'text':'before partial','generated_tokens':2,'truncated':False} for r in rows}
                model=SimpleNamespace(eval=lambda:None,gradient_checkpointing_disable=lambda:None,config=SimpleNamespace(use_cache=False))
                ns.update(np=np,pd=pd,csv=csv,Counter=Counter,random=random,tqdm=lambda items,**kw:items,display=lambda *a:None,
                          eval_model=model,trainer=SimpleNamespace(model=model),EVAL_ADAPTER_DIR='mock',ADAPTER_DIR='mock',ADAPTER_STEP=1,
                          validation_callback=SimpleNamespace(best_step=1),EVAL_JUDGE_MODEL=ns['GROQ_MODEL'],
                          generate_sample_response=lambda *a,**kw:{'text':'after full','generated_tokens':3,'truncated':False},
                          tokenizer=object(),EVALUATE_BENCHMARK_BASELINE=True,baseline_outputs=baseline,
                          TRAIN_PROMPTS=set(),overlaps_training_prompt=lambda text:False,RESULTS_DIR=directory,EVAL_RESULTS_DIR=directory,
                          hard_result=lambda text,spec:{'partial_score':1. if text.endswith('full') else .5,'all_hard_passed':text.endswith('full'),'details':[{'id':'hard','passed':text.endswith('full')}]})
                with redirect_stdout(StringIO()):_,metrics=ns['run_benchmark']('seen',rows)
                outputs=json.loads((Path(directory)/'eval_outputs_seen.json').read_text())
                self.assertEqual(len(calls),4)
                for record in outputs:
                    for stage in ('before','after'):
                        detail=record['soft_judge_'+stage]
                        self.assertEqual(detail['provider'],'groq')
                        self.assertEqual(detail['model'],'llama-3.3-70b-versatile')
                        self.assertTrue(all(score['reason'] for score in detail['scores']))
                table=pd.read_csv(Path(directory)/'eval_soft_criteria_seen.csv')
                self.assertEqual(set(table.judge_provider),{'groq'})
                fixtures.definitions(nb,9 if 'colab' in name else 12,ns)
                with redirect_stdout(StringIO()):report=ns['run_judge_calibration'](directory,max_responses=4,providers=['groq'])
                self.assertEqual(report['human_calibration_status'],'pending')
                self.assertIsNone(report['agreement_with_human']['groq']['overall']['agreement'])
                self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['groq'],len(calls))
                for provider in ('gemma','gemini','openrouter'):
                    self.assertEqual(ns['_JUDGE_PROVIDER_CALLS'][provider],0)

    def test_missing_key_and_hard_only_preflight(self):
        for name in FILES:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                ns,_,_=groq_namespace(name,directory)
                fake=types.ModuleType('kaggle_secrets');fake.UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda _:None)
                colab=types.ModuleType('google.colab');colab.userdata=SimpleNamespace(get=lambda _:None)
                google=types.ModuleType('google');google.colab=colab
                with patch.dict(sys.modules,{'kaggle_secrets':fake,'google':google,'google.colab':colab}):
                    with self.assertRaisesRegex(RuntimeError,'GROQ_API_KEY'):ns['_get_groq_client']()
                if name!=fixtures.FILES[2]:
                    ns['REWARD_MODE']='hard_only';ns['run_soft_judge_preflight']()
                    ns['REWARD_MODE']='hybrid_gated';ns['RUN_JUDGE_PREFLIGHT']=False;ns['run_soft_judge_preflight']()

    def test_resume_requires_same_groq_config_and_baseline_unchanged(self):
        for name in FILES[:1]+[fixtures.FILES[3]]:
            nb=fixtures.notebook(name);ns=fixtures.judge_namespace(name);ns['TrainerCallback']=object
            fixtures.definitions(nb,4,ns);fixtures.definitions(nb,9,ns)
            with redirect_stdout(StringIO()):exec(''.join(nb['cells'][8]['source']),ns)
            rows=[ns['load_rows'](fixtures.ROOT/p) for p in fixtures.DATA]
            fit,val,seen,unseen,manifest=ns['prepare_data_splits'](*rows)
            context={'model':ns['MODEL_NAME'],'seed':42,'max_prompt_length':2048,'max_completion_length':512,
                     'chat_template_kwargs':ns['CHAT_TEMPLATE_KWARGS'],'split_signature':'mock-split'}
            ns.update(TRAIN_ROWS=fit,VALIDATION_ROWS=val[:32],TEST_SEEN_ROWS=seen,TEST_UNSEEN_ROWS=unseen,
                      SPLIT_MANIFEST=manifest,SPLIT_SIGNATURE='mock-split',PREPARED_CONTEXT=context,
                      MAX_PROMPT_LENGTH=2048,EVALUATE_BENCHMARK_BASELINE=True,REWARD_MODE='hybrid_gated',
                      JUDGE_PROVIDER='groq',JUDGE_BASE_URL=ns['GROQ_BASE_URL'],JUDGE_MODEL=ns['GROQ_MODEL'],
                      MODEL_DTYPE='mock-fp32',torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1)),
                      importlib=importlib,BASELINE_IMPORT_FILE='',train_dataset=fit)
            fake_datasets=SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda values:values))
            with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'datasets':fake_datasets}), \
                 patch.object(importlib.metadata,'version',return_value='mock-version'):
                ns.update(RUN_DIR=directory,RESULTS_DIR=str(Path(directory)/'results'),BASELINE_FILE=str(Path(directory)/'results/baseline_outputs.json'))
                Path(ns['RESULTS_DIR']).mkdir()
                cache={ns['baseline_key']('validation',r):{'text':'baseline','generated_tokens':1,'truncated':False} for r in ns['VALIDATION_ROWS']}
                ns['save_json'](ns['BASELINE_FILE'],cache);before=Path(ns['BASELINE_FILE']).read_bytes()
                first,config,_,_=ns['prepare_training_attempt']()
                self.assertEqual(config['groq_request_config']['response_format'],{'type':'json_object'})
                self.assertEqual(config['judge_model'],'llama-3.3-70b-versatile')
                self.assertNotIn('api_key',config['groq_request_config'])
                checkpoint=first/'checkpoints/checkpoint-50';checkpoint.mkdir(parents=True);(checkpoint/'trainer_state.json').write_text('{}')
                ns['RESUME_CHECKPOINT']=str(checkpoint)
                _,resumed,_,_=ns['prepare_training_attempt']();self.assertEqual(config,resumed)
                ns['GROQ_MIN_INTERVAL_SECONDS']=12
                with self.assertRaises(ValueError):ns['prepare_training_attempt']()
                self.assertEqual(before,Path(ns['BASELINE_FILE']).read_bytes())


if __name__=='__main__':
    unittest.main(verbosity=2)
