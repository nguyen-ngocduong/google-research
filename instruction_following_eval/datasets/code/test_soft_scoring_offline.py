"""Offline regression checks. No model download, training, real secrets or API.

Run: python3 datasets/code/test_soft_scoring_offline.py
"""
import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import soft_constraints_policy as policy

ROOT=Path(__file__).resolve().parents[2]
FILES=['baseline_qwen3_5_0_8b_colab.ipynb','train_qwen3_5_0_8b_colab.ipynb',
       'eval_qwen3_5_0_8b_colab.ipynb','training/finetune_qwen3_5_0_8b_kaggle.ipynb']
DATA=['datasets/grpo/train_seen.jsonl','datasets/eval/test_seen.jsonl','datasets/eval/test_unseen.jsonl']


def notebook(name):
    return json.loads((ROOT/name).read_text())


def definitions(nb,index,ns):
    nodes=[n for n in ast.parse(''.join(nb['cells'][index]['source'])).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<offline notebook functions>','exec'),ns)


def common_namespace():
    ns={'json':json,'hashlib':hashlib,'os':os,'Path':Path,'copy':copy,'SEED':42,
        'MODEL_NAME':'Qwen/Qwen3.5-0.8B','MAX_COMPLETION_LENGTH':512,
        'CHAT_TEMPLATE_KWARGS':{'enable_thinking':False},'VALIDATION_FRACTION':.1,'TEST_FRACTION':.2}
    exec((ROOT/'datasets/code/soft_constraints_policy.py').read_text(),ns)
    return ns


class StatusError(Exception):
    def __init__(self,status,delay=0):
        self.status_code=status
        self.response=SimpleNamespace(headers={'retry-after':str(delay)})


class ConnectionError(Exception):pass
class TimeoutError(Exception):pass


def fake_response(criteria, score=1):
    text=json.dumps({'scores':[{'id':c['id'],'score':score,'reason':'Thực hiện đủ yêu cầu.' if score else 'Thiếu yêu cầu.'} for c in criteria]})
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=text))])


def client(fn):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fn)))


def criterion():
    return policy.normalize_soft_constraints([{'id':'style_and_tone:test','category':'style_and_tone',
            'description':'Dùng giọng lịch sự.','rubric':'1 lịch sự; 0 xúc phạm.',
            'verification_status':'PASS','verified_by':'offline_fixture'}])[0][0]


def judge_namespace(name):
    nb=notebook(name);ns=common_namespace();definitions(nb,3,ns);definitions(nb,6,ns)
    ns.update(os=SimpleNamespace(path=os.path,environ={}),threading=threading,deque=deque,math=math,re=re,
              ThreadPoolExecutor=ThreadPoolExecutor,Future=Future,APIStatusError=StatusError,
              APIConnectionError=ConnectionError,APITimeoutError=TimeoutError,
              JUDGE_PROVIDER='gemma',JUDGE_RUBRIC_VERSION='soft-v2',SOFT_REWARD_GATE='all_hard',
              GEMMA_BASE_URL='https://mock-gemma.invalid/v1',GEMMA_MODEL='google/gemma-4-31B-it',
              GEMMA_API_KEY_SECRET='GEMMA_API_KEY',GEMMA_MAX_REQUESTS=0,GEMMA_WINDOW_SECONDS=15,
              GEMINI_BASE_URL='https://mock-google.invalid/v1',GEMINI_MODEL='gemini-3.5-flash',
              JUDGE_TIMEOUT_SECONDS=60,JUDGE_MAX_TOKENS=4096,JUDGE_MAX_WORKERS=4,
              JUDGE_MAX_ATTEMPTS=3,JUDGE_MAX_REQUESTS=4,JUDGE_WINDOW_SECONDS=15,
              JUDGE_KEY_NAMES=('GEMINI_API_KEY','GEMINI_API_KEY_1'),_JUDGE_CLIENTS=None,
              _GEMINI_CLIENT_CONFIG=None,_GEMMA_CLIENT=None,_GEMMA_CLIENT_CONFIG=None,
              _GEMMA_LIMITER=None,_JUDGE_STATE_LOCK=threading.Lock(),_SOFT_JUDGE_CACHE={},
              _JUDGE_INFLIGHT={},_JUDGE_CALL_COUNT=0,_JUDGE_ERROR_COUNT=0,
              _JUDGE_ACTIVE_KEY_INDEX=0,_JUDGE_SWITCH_COUNT=0,_JUDGE_CALLS_PER_KEY=[0,0],
              _JUDGE_PROVIDER_CALLS={'gemma':0,'gemini':0},time=SimpleNamespace(sleep=lambda _:None),
              OPENROUTER_BASE_URL='https://openrouter.ai/api/v1',OPENROUTER_MODEL='google/gemma-4-31b-it:free',
              OPENROUTER_API_KEY_SECRET='OPENROUTER_API_KEY',OPENROUTER_MAX_REQUESTS=4,
              OPENROUTER_WINDOW_SECONDS=15.,OPENROUTER_REASONING='off',
              GROQ_BASE_URL='https://api.groq.com/openai/v1',GROQ_MODEL='llama-3.3-70b-versatile',
              GROQ_API_KEY_SECRET='GROQ_API_KEY',GROQ_MAX_REQUESTS=4,GROQ_WINDOW_SECONDS=60.,
              GROQ_MIN_INTERVAL_SECONDS=10.,GROQ_MAX_RETRY_WAIT_SECONDS=120.,RUN_JUDGE_PREFLIGHT=True,
              _GROQ_CLIENT=None,_GROQ_CLIENT_CONFIG=None,
              _OPENROUTER_CLIENT=None,_OPENROUTER_CLIENT_CONFIG=None,
              GLOBAL_REWARD_CONFIG={'mode':'hybrid_gated','w_hard':1.,'w_soft':.25,
                                   'hard_partial_weight':.7,'hard_bonus_all_pass':.3,'soft_gate':'all_hard'},
              _SOFT_REWARD_STATS={'completions':0,'eligible':0,'score_sum':0.,'zero_signal_batches':0})
    ns['_read_optional_gemma_key']=lambda:''
    ns['OpenAI']=lambda **kw:client(lambda **kwargs:fake_response(json.loads(kwargs['messages'][1]['content'])['criteria']))
    return ns


class SoftDataTests(unittest.TestCase):
    def test_reject_and_unverified_cannot_be_overridden(self):
        good=criterion()
        for status in ('REJECT','UNVERIFIED','PASS (default)'):
            original=dict(good,verification_status=status)
            approved,excluded=policy.select_row_soft_constraints({'soft':json.dumps([good]),'soft_constraints':{'constraints':[original]}})
            self.assertFalse(approved);self.assertEqual(len(excluded),1)
        missing=dict(good);missing.pop('verification_status')
        self.assertFalse(policy.normalize_soft_constraints([missing])[0])
        with self.assertRaises(ValueError):policy.normalize_soft_constraints([good,good])
        with self.assertRaises(ValueError):policy.normalize_soft_constraints([dict(good,rubric='')])

    def test_dataset_and_backup_integrity(self):
        report=json.loads((ROOT/'datasets/soft_audit/repair_report_soft_v2.json').read_text())
        self.assertEqual(len(report['changed_prompts']),11)
        self.assertEqual(sum(len(r.get('excluded',[])) for r in report['quarantine']),13)
        self.assertEqual([report['summary']['rows_after'][n] for n in DATA],[499,499,96])
        rows=[]
        for name in DATA:
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),report['output_sha256'][name])
            original=ROOT/'datasets/soft_audit/original_v1'/Path(name).name
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),report['input_sha256'][name])
            values=[json.loads(l) for l in (ROOT/name).read_text().splitlines()]
            rows.append(values)
            for row in values:
                accepted,excluded=policy.select_row_soft_constraints(row)
                self.assertTrue(accepted);self.assertFalse(excluded)
                self.assertEqual(row['reward_spec'],policy.soft_reward_spec())
                self.assertTrue(all(c['evaluation_guidance'] for c in accepted))
        seen={r['key']:r for r in rows[1]}
        for row in rows[0]:
            self.assertEqual(row['prompt'],seen[row['key']]['messages'])
            self.assertEqual(row['soft'],seen[row['key']]['soft'])
        categories={c['category'] for row in rows[2] for c in json.loads(row['soft'])}
        self.assertEqual(categories,set(policy.UNSEEN_SOFT_CATEGORIES))

    def test_verification_failure_and_id_mapping(self):
        path=ROOT/'datasets/code/generate_soft_constraints_vi.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_soft_constraint_with_gemini')
        ns={'json':json,'List':List,'Dict':Dict,'Any':Any}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        originals=[dict(criterion(),id='a'),dict(criterion(),id='b')]
        for response in (None,'not JSON','[]','[{"id":"a","verification_status":"PASS"}]',
                         '[{"id":"a","verification_status":"PASS","verification_reason":"ok"},{"id":"a","verification_status":"PASS","verification_reason":"ok"}]'):
            ns['call_gemini_verify']=lambda *a,**kw:response
            out=ns['verify_soft_constraint_with_gemini']('mock','task',[],copy.deepcopy(originals))
            self.assertTrue(all(c['verification_status']=='UNVERIFIED' for c in out))
        ns['call_gemini_verify']=lambda *a,**kw:json.dumps([{'id':'b','verification_status':'REJECT','verification_reason':'Conflict'},
                                                          {'id':'a','verification_status':'PASS','verification_reason':'Compatible'}])
        out=ns['verify_soft_constraint_with_gemini']('mock','task',[],copy.deepcopy(originals))
        self.assertEqual([(c['id'],c['verification_status']) for c in out],[('a','PASS'),('b','REJECT')])
        out=ns['verify_soft_constraint_with_gemini']('', 'task',[],copy.deepcopy(originals))
        self.assertTrue(all(c['verification_status']=='UNVERIFIED' for c in out))

    def test_packager_all_rejected_preserves_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);source=path/'input.json'
            source.write_text(json.dumps([{'key':'fixture','base_instruction':'task',
                                           'soft_constraints':{'constraints':[dict(criterion(),verification_status='REJECT')]}}]))
            result=subprocess.run([sys.executable,str(ROOT/'datasets/code/package_final_dataset_vi.py'),
                                   '--input-dir',str(source),'--output-dir',str(path/'out')],cwd=path,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            quarantine=list((path/'out').glob('soft_quarantine_*.json'))
            self.assertEqual(len(quarantine),1)
            self.assertTrue(json.loads(quarantine[0].read_text())[0]['whole_record_excluded'])
            self.assertFalse(list((path/'out').glob('module_final_batch_*.json')))

    def test_migration_is_reproducible_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in DATA:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/'datasets/soft_audit/original_v1'/Path(name).name,target)
            command=[sys.executable,str(ROOT/'datasets/code/repair_soft_constraints_vi.py'),'--root',str(root),'--apply']
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            for name in DATA:self.assertEqual((root/name).read_bytes(),(ROOT/name).read_bytes())
            before=(root/'datasets/soft_audit/repair_report_soft_v2.json').read_bytes()
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertIn('không thay đổi thêm',second.stdout)
            self.assertEqual(before,(root/'datasets/soft_audit/repair_report_soft_v2.json').read_bytes())


class NotebookTests(unittest.TestCase):
    def test_kaggle_complete_config_and_judge_cell(self):
        nb=notebook(FILES[3])
        ns=common_namespace();definitions(nb,3,ns)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['train_seen.jsonl','test_seen.jsonl','test_unseen.jsonl','verifier_engine_vi.py']:
                p=root/'input'/name;p.parent.mkdir(exist_ok=True);p.write_text('fixture')
            real_path=type(Path())
            def mapped(value):
                text=str(value)
                return root/text[len('/kaggle/'):] if text.startswith('/kaggle/') else real_path(value)
            class VirtualPath(real_path):
                def __new__(cls,*parts):
                    if len(parts)==1 and str(parts[0]).startswith('/kaggle/'):
                        return super().__new__(cls,mapped(parts[0]))
                    return super().__new__(cls,*parts)
                def __init__(self,*parts):
                    if real_path.__init__ is not object.__init__:
                        if len(parts)==1 and str(parts[0]).startswith('/kaggle/'):
                            super().__init__(mapped(parts[0]))
                        else:
                            super().__init__(*parts)
            fake_os=types.ModuleType('os');fake_os.__dict__.update(os.__dict__)
            fake_os.environ={}
            fake_os.makedirs=lambda path,**kw:os.makedirs(mapped(path),**kw)
            # Execute all configuration statements, including imports and validation,
            # with filesystem writes redirected away from /kaggle/working.
            with patch.dict(sys.modules,{'os':fake_os}),patch.object(pathlib,'Path',VirtualPath),redirect_stdout(StringIO()):
                exec(''.join(nb['cells'][2]['source']).replace("JUDGE_PROVIDER = 'groq'","JUDGE_PROVIDER = 'openrouter'"),ns)
            self.assertEqual(ns['JUDGE_PROVIDER'],'openrouter')
            self.assertEqual(ns['JUDGE_MODEL'],'google/gemma-4-31b-it:free')
            self.assertEqual(ns['JUDGE_BASE_URL'],'https://openrouter.ai/api/v1')
            self.assertEqual(ns['SOFT_REWARD_GATE'],'all_hard')
            self.assertEqual(ns['JUDGE_RUBRIC_VERSION'],'soft-v2')
            self.assertTrue(ns['JUDGE_CACHE_FILE'].endswith('soft_judge_cache_v2.jsonl'))
            # Entire judge cell body (imports supplied as mocks), not just functions.
            tree=ast.parse(''.join(nb['cells'][6]['source']))
            statements=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
            calls=[];secret_reads=[];created=[];now=[0.]
            def get_secret(name):
                secret_reads.append(name)
                if name!='OPENROUTER_API_KEY':raise AssertionError('Unexpected Google/company secret access')
                return 'mock-openrouter-secret'
            fake_secrets=types.ModuleType('kaggle_secrets')
            fake_secrets.UserSecretsClient=lambda:SimpleNamespace(get_secret=get_secret)
            def create(**kw):
                calls.append(kw)
                return fake_response(json.loads(kw['messages'][1]['content'])['criteria'])
            def factory(**kw):
                created.append(kw);return client(create)
            ns.update(os=fake_os,Path=real_path,math=math,re=re,RESULTS_DIR=str(root/'working'),
                      JUDGE_CACHE_FILE=str(root/'working/soft_judge_cache_v2.jsonl'),
                      OpenAI=factory,APIStatusError=StatusError,APIConnectionError=ConnectionError,
                      APITimeoutError=TimeoutError,threading=threading,deque=deque,Future=Future,
                      ThreadPoolExecutor=ThreadPoolExecutor,
                      time=SimpleNamespace(monotonic=lambda:now[0],sleep=lambda delay:now.__setitem__(0,now[0]+delay)))
            (root/'working').mkdir(exist_ok=True)
            # Write a legacy scalar cache: it must be ignored.
            (root/'working/gemini_judge_cache.jsonl').write_text('{"key":"old","score":1}\n')
            with patch.dict(sys.modules,{'kaggle_secrets':fake_secrets}),redirect_stdout(StringIO()):
                exec(compile(ast.Module(body=statements,type_ignores=[]),'<full Kaggle judge cell>','exec'),ns)
                self.assertFalse(calls)
                self.assertEqual(secret_reads,['OPENROUTER_API_KEY'])
                self.assertEqual(created[0]['max_retries'],0)
                self.assertEqual(created[0]['base_url'],'https://openrouter.ai/api/v1')
                criteria=[criterion()]
                result=ns['evaluate_single_soft_details']('answer','task',criteria)
                self.assertEqual(result['provider'],'openrouter')
                self.assertEqual(result['score'],1.)
                self.assertTrue(result['scores'][0]['reason'])
                self.assertEqual(calls[0]['extra_body'],{'reasoning':{'exclude':True,'enabled':False}})
                self.assertNotIn('reasoning_effort',calls[0]);self.assertNotIn('response_format',calls[0])
                self.assertNotIn('old',ns['_SOFT_JUDGE_CACHE'])
                key=ns['_compute_cache_key']('task','answer',criteria,'openrouter')
                limiter=ns['_OPENROUTER_LIMITER'];client_before=ns['_OPENROUTER_CLIENT']
                exec(compile(ast.Module(body=statements,type_ignores=[]),'<rerun Kaggle judge cell>','exec'),ns)
                self.assertIs(limiter,ns['_OPENROUTER_LIMITER'])
                self.assertIs(client_before,ns['_OPENROUTER_CLIENT'])
                self.assertEqual(ns['evaluate_single_soft_details']('answer','task',criteria)['score'],1.)
                self.assertEqual(len(calls),1)
                self.assertEqual(secret_reads,['OPENROUTER_API_KEY'])
                ns['OPENROUTER_MODEL']='nvidia/nemotron-3-ultra-550b-a55b:free'
                self.assertNotEqual(key,ns['_compute_cache_key']('task','answer',criteria,'openrouter'))
                ns['OPENROUTER_MODEL']='google/gemma-4-31b-it:free';ns['OPENROUTER_REASONING']='low'
                self.assertNotEqual(key,ns['_compute_cache_key']('task','answer',criteria,'openrouter'))
                self.assertEqual(ns['_openrouter_request_body'](),{'reasoning':{'exclude':True,'effort':'low'}})
                self.assertEqual(ns['_JUDGE_PROVIDER_CALLS'],{'gemma':0,'gemini':0,'openrouter':1,'groq':0})

    def test_openrouter_errors_and_rate_limit(self):
        ns=judge_namespace(FILES[3]);criteria=[criterion()]
        ns.update(JUDGE_PROVIDER='openrouter',_JUDGE_PROVIDER_CALLS={'gemma':0,'gemini':0,'openrouter':0})
        # Missing required OpenRouter secret must not fall back to a dummy token.
        missing=types.ModuleType('kaggle_secrets')
        missing.UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda name:(_ for _ in ()).throw(KeyError(name)))
        with patch.dict(sys.modules,{'kaggle_secrets':missing}):
            with self.assertRaisesRegex(RuntimeError,'OPENROUTER_API_KEY'):ns['_get_openrouter_client']()
        now=[0.];starts=[];waits=[]
        def sleep(delay):waits.append(delay);now[0]+=delay
        limiter=ns['SlidingWindowRateLimiter'](4,15,clock=lambda:now[0],sleeper=sleep)
        class Traced:
            def acquire(self):limiter.acquire();starts.append(now[0])
        attempts=[]
        def rate_limited(**kw):
            attempts.append(kw)
            if len(attempts)==1:raise StatusError(429,23)
            return fake_response(criteria)
        ns.update(_OPENROUTER_CLIENT=client(rate_limited),
                  _OPENROUTER_CLIENT_CONFIG=(ns['OPENROUTER_BASE_URL'],60,'OPENROUTER_API_KEY'),
                  _OPENROUTER_LIMITER=Traced(),time=SimpleNamespace(sleep=sleep))
        with redirect_stdout(StringIO()):
            for i in range(7):self.assertEqual(ns['_request_soft_score'](str(i),'task',criteria)['score'],1.)
        self.assertIn(23,waits)
        for instant in starts:self.assertLessEqual(sum(instant-15<t<=instant for t in starts),4)
        for code in (400,401,402,403,404):
            ns['_OPENROUTER_CLIENT']=client(lambda **kw:(_ for _ in ()).throw(StatusError(code)))
            before=ns['_JUDGE_PROVIDER_CALLS']['openrouter']
            with self.assertRaisesRegex(RuntimeError,f'HTTP {code}'):ns['_request_soft_score']('x','task',criteria)
            self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['openrouter'],before+1)
        daily=StatusError(429);daily.body={'error':{'message':'Daily free-model limit exceeded'}}
        ns['_OPENROUTER_CLIENT']=client(lambda **kw:(_ for _ in ()).throw(daily))
        with tempfile.TemporaryDirectory() as directory:
            ns['JUDGE_CACHE_FILE']=str(Path(directory)/'cache.jsonl')
            before=ns['_JUDGE_PROVIDER_CALLS']['openrouter']
            with self.assertRaisesRegex(RuntimeError,'quota ngày'):ns['evaluate_single_soft_details']('x','task',criteria)
            self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['openrouter'],before+1)
            self.assertFalse(ns['_SOFT_JUDGE_CACHE']);self.assertFalse(ns['_JUDGE_INFLIGHT'])
            self.assertFalse(Path(ns['JUDGE_CACHE_FILE']).exists())
            # Incomplete reasoning output consumes bounded attempts and cannot be a score.
            ns['_OPENROUTER_CLIENT']=client(lambda **kw:SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(content=''))]))
            before=ns['_JUDGE_PROVIDER_CALLS']['openrouter']
            with redirect_stdout(StringIO()),self.assertRaises(RuntimeError):ns['evaluate_single_soft_details']('x','task',criteria)
            self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['openrouter'],before+3)
            self.assertFalse(Path(ns['JUDGE_CACHE_FILE']).exists())
        self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['gemma'],0);self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['gemini'],0)


    def test_resume_preserves_policy_and_baseline_cache(self):
        for name in (FILES[1],FILES[3]):
            ns=judge_namespace(name);nb=notebook(name)
            ns['TrainerCallback']=type('MockTrainerCallback',(),{})
            definitions(nb,4,ns);definitions(nb,9,ns)
            with redirect_stdout(StringIO()):
                exec(''.join(nb['cells'][8]['source']),ns)
            rows=[ns['load_rows'](ROOT/p) for p in DATA]
            fit,val,seen,unseen,manifest=ns['prepare_data_splits'](*rows)
            context={'model':ns['MODEL_NAME'],'seed':42,'max_prompt_length':2048,'max_completion_length':512,
                     'chat_template_kwargs':ns['CHAT_TEMPLATE_KWARGS'],'split_signature':'mock-split'}
            ns.update(TRAIN_ROWS=fit,VALIDATION_ROWS=val[:32],TEST_SEEN_ROWS=seen,TEST_UNSEEN_ROWS=unseen,
                      SPLIT_MANIFEST=manifest,SPLIT_SIGNATURE='mock-split',PREPARED_CONTEXT=context,
                      MAX_PROMPT_LENGTH=2048,EVALUATE_BENCHMARK_BASELINE=True,REWARD_MODE='hybrid_gated',
                      JUDGE_BASE_URL=ns['GEMMA_BASE_URL'],JUDGE_MODEL=ns['GEMMA_MODEL'],MODEL_DTYPE='mock-fp32',
                      torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1)),importlib=importlib,
                      BASELINE_IMPORT_FILE='',train_dataset=fit)
            fake_datasets=SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda values:values))
            with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'datasets':fake_datasets}), \
                 patch.object(importlib.metadata,'version',return_value='mock-version'):
                ns.update(RUN_DIR=directory,RESULTS_DIR=str(Path(directory)/'results'),
                          BASELINE_FILE=str(Path(directory)/'results/baseline_outputs.json'))
                Path(ns['RESULTS_DIR']).mkdir()
                cache={ns['baseline_key']('validation',r):{'text':'baseline','generated_tokens':1,'truncated':False} for r in ns['VALIDATION_ROWS']}
                ns['save_json'](ns['BASELINE_FILE'],cache)
                before=Path(ns['BASELINE_FILE']).read_bytes()
                first,config,_,loaded=ns['prepare_training_attempt']()
                self.assertEqual(config['judge_rubric_version'],'soft-v2')
                self.assertEqual(config['soft_reward_spec'],policy.soft_reward_spec())
                self.assertEqual(cache,loaded)
                checkpoint=first/'checkpoints/checkpoint-50';checkpoint.mkdir(parents=True)
                (checkpoint/'trainer_state.json').write_text('{}')
                ns['RESUME_CHECKPOINT']=str(checkpoint)
                _,resumed,previous,_=ns['prepare_training_attempt']()
                self.assertEqual(config,resumed);self.assertEqual(previous,first)
                for variable,value in [('SOFT_REWARD_GATE','partial_hard'),('JUDGE_RUBRIC_VERSION','soft-v3')]:
                    original=ns[variable];ns[variable]=value
                    with self.assertRaisesRegex(ValueError,'Resume'):ns['prepare_training_attempt']()
                    ns[variable]=original
                self.assertEqual(Path(ns['BASELINE_FILE']).read_bytes(),before)
                if 'kaggle' in name:
                    ns.update(JUDGE_PROVIDER='openrouter',JUDGE_MODEL=ns['OPENROUTER_MODEL'],
                              JUDGE_BASE_URL=ns['OPENROUTER_BASE_URL'],RESUME_CHECKPOINT='')
                    attempt,config,_,_=ns['prepare_training_attempt']()
                    self.assertEqual(config['openrouter_request_config']['extra_body'],
                                     {'reasoning':{'exclude':True,'enabled':False}})
                    checkpoint=attempt/'checkpoints/checkpoint-50';checkpoint.mkdir(parents=True)
                    (checkpoint/'trainer_state.json').write_text('{}')
                    ns['RESUME_CHECKPOINT']=str(checkpoint)
                    self.assertEqual(ns['prepare_training_attempt']()[1],config)
                    for variable,value in [('OPENROUTER_REASONING','low'),
                                           ('OPENROUTER_MODEL','nvidia/nemotron-3-ultra-550b-a55b:free'),
                                           ('OPENROUTER_BASE_URL','https://different.invalid/v1'),
                                           ('OPENROUTER_MAX_REQUESTS',2)]:
                        original=ns[variable];ns[variable]=value
                        with self.assertRaisesRegex(ValueError,'Resume'):ns['prepare_training_attempt']()
                        ns[variable]=original
                    self.assertEqual(Path(ns['BASELINE_FILE']).read_bytes(),before)

    def test_syntax_shared_splits_and_input_guard(self):
        reference=None
        for name in FILES:
            nb=notebook(name)
            self.assertEqual(len({c['id'] for c in nb['cells']}),len(nb['cells']))
            for cell in nb['cells']:
                if cell['cell_type']=='code':
                    ast.parse(''.join(cell['source']))
                    self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
            ns=common_namespace();definitions(nb,3,ns)
            rows=[ns['load_rows'](ROOT/path) for path in DATA]
            fit,val,seen,unseen,manifest=ns['prepare_data_splits'](*rows)
            if reference is None:reference=manifest
            else:self.assertEqual(manifest,reference)
            self.assertEqual(len(fit)+len(val)+len(seen),499)
            self.assertEqual(len(unseen),96)
            with self.assertRaisesRegex(ValueError,'Upload ba dataset'):
                old=json.loads((ROOT/'datasets/soft_audit/original_v1/train_seen.jsonl').read_text().splitlines()[0])
                ns['normalize_row'](old)
            # Changed prompts cannot hit old baseline hashes; unchanged ones can.
            original_unseen={r['key']:r for r in map(json.loads,(ROOT/'datasets/soft_audit/original_v1/test_unseen.jsonl').read_text().splitlines())}
            changed=next(r for r in unseen if r['key']=='alpaca-gpt4_0034')
            old=dict(original_unseen[changed['key']]);old['prompt']=ns['prompt_messages'](old['prompt'])
            self.assertNotEqual(ns['baseline_key']('unseen',changed),ns['baseline_key']('unseen',old))
        print('Offline split counts:',reference['train_count'],reference['validation_count'],reference['seen_test_count'],reference['unseen_test_count'])

    def test_binary_parser_details_and_cache(self):
        for name in FILES[1:]:
            ns=judge_namespace(name);criteria=[criterion()]
            result=ns['_parse_judge_scores'](fake_response(criteria).choices[0].message.content,criteria)
            self.assertEqual(result['score'],1);self.assertEqual(result['scores'][0]['category'],'style_and_tone')
            for score in (True,-1,.5,2):
                with self.assertRaises(ValueError):ns['_parse_judge_scores'](json.dumps({'scores':[{'id':criteria[0]['id'],'score':score,'reason':'x'}]}),criteria)
            for reason in ('','x'*401):
                with self.assertRaises(ValueError):ns['_parse_judge_scores'](json.dumps({'scores':[{'id':criteria[0]['id'],'score':1,'reason':reason}]}),criteria)
            with tempfile.TemporaryDirectory() as d:
                ns['JUDGE_CACHE_FILE']=str(Path(d)/'judge.jsonl')
                self.assertEqual(ns['evaluate_single_soft_llm']('answer','task',criteria),1)
                self.assertEqual(ns['evaluate_single_soft_llm']('answer','task',criteria),1)
                self.assertEqual(ns['_JUDGE_CALL_COUNT'],1)
                detail=ns['evaluate_single_soft_details']('answer','task',criteria)
                self.assertTrue(detail['scores'][0]['reason'])
                gemma_key=ns['_compute_cache_key']('task','answer',criteria,'gemma')
                self.assertNotEqual(gemma_key,ns['_compute_cache_key']('task','answer',criteria,'gemini'))
                self.assertIsNone(ns['evaluate_single_soft_details']('answer','task',[])['score'])
                self.assertEqual(ns['evaluate_single_soft_llm']('answer','task',[]),0)
                self.assertEqual(ns['evaluate_single_soft_llm']('','task',criteria),0)
                cached=json.loads(Path(ns['JUDGE_CACHE_FILE']).read_text())
                self.assertEqual(cached['result']['rubric_version'],'soft-v2')
                def failure(*a,**kw):raise RuntimeError('Mock quota exhausted')
                ns['_request_soft_score']=failure
                with self.assertRaises(RuntimeError):ns['evaluate_single_soft_llm']('new answer','task',criteria)
                self.assertFalse(ns['_JUDGE_INFLIGHT']);self.assertEqual(len(ns['_SOFT_JUDGE_CACHE']),1)

    def test_concurrent_dedup_and_reward_gate(self):
        ns=judge_namespace(FILES[1]);criteria=[criterion()]
        calls=[];barrier=threading.Barrier(4)
        def request(*args):
            calls.append(1);time.sleep(.03)
            return {'score':1.,'scores':[{'id':criteria[0]['id'],'category':'style_and_tone','score':1,'reason':'Đạt.'}]}
        def concurrent(_):
            barrier.wait(timeout=3)
            return ns['evaluate_single_soft_llm']('duplicate','task',criteria)
        with tempfile.TemporaryDirectory() as d:
            ns.update(JUDGE_CACHE_FILE=str(Path(d)/'judge.jsonl'),_request_soft_score=request)
            with ThreadPoolExecutor(max_workers=4) as pool:self.assertEqual(list(pool.map(concurrent,range(4))),[1.]*4)
            self.assertEqual(len(calls),1)
            ns['hard_result']=lambda text,spec:{'partial_score':1. if text=='full' else .5,'all_hard_passed':text=='full'}
            args=dict(prompts=['task']*3,completions=['full','partial','partial'],verifier=['v']*3,soft=[criteria]*3)
            self.assertEqual(ns['reward_soft'](**args),[.25,0.,0.])
            stats=ns['soft_reward_statistics']();self.assertEqual(stats['eligible'],1);self.assertEqual(stats['eligible_fraction'],1/3)
            self.assertEqual(ns['reward_hard'](**args),[1.,.35,.35])
            ns['GLOBAL_REWARD_CONFIG']['soft_gate']='partial_hard'
            self.assertEqual(ns['reward_soft'](**args),[.25,.25,.25])

    def test_transport_retries_failover_and_shared_rate_limit(self):
        for name in FILES[1:]:
            ns=judge_namespace(name);criteria=[criterion()];waits=[];attempts=[]
            ns['time']=SimpleNamespace(sleep=lambda delay:waits.append(delay))
            def gemma(**kw):
                attempts.append(kw)
                if len(attempts)==1:raise StatusError(503,5)
                return fake_response(criteria)
            ns['_GEMMA_CLIENT']=client(gemma)
            ns['_GEMMA_CLIENT_CONFIG']=(ns['GEMMA_BASE_URL'],60,'GEMMA_API_KEY')
            self.assertEqual(ns['_request_soft_score']('a','task',criteria)['score'],1)
            self.assertEqual(waits,[5]);self.assertEqual(len(attempts),2)
            self.assertEqual(set(attempts[0]),{'model','messages','temperature','max_tokens','timeout'})
            for code in (400,401,403,404):
                def fatal(**kw):raise StatusError(code)
                ns['_GEMMA_CLIENT']=client(fatal)
                before=ns['_JUDGE_PROVIDER_CALLS']['gemma']
                with self.assertRaises(RuntimeError):ns['_request_soft_score']('a','task',criteria)
                self.assertEqual(ns['_JUDGE_PROVIDER_CALLS']['gemma'],before+1)
            clock=[0.];starts=[]
            def sleep(delay):clock[0]+=delay
            limiter=ns['SlidingWindowRateLimiter'](4,15,clock=lambda:clock[0],sleeper=sleep)
            class Trace:
                def acquire(self):limiter.acquire();starts.append(clock[0])
            def primary(**kw):raise StatusError(429)
            def backup(**kw):return fake_response(criteria)
            ns.update(JUDGE_PROVIDER='gemini',_JUDGE_CLIENTS=[client(primary),client(backup)],
                      _GEMINI_CLIENT_CONFIG=(ns['GEMINI_BASE_URL'],60),_JUDGE_LIMITER=Trace(),time=SimpleNamespace(sleep=sleep))
            for i in range(9):self.assertEqual(ns['_request_soft_score'](str(i),'task',criteria)['score'],1)
            self.assertEqual(ns['_JUDGE_CALLS_PER_KEY'],[1,9]);self.assertEqual(ns['_JUDGE_SWITCH_COUNT'],1)
            for instant in starts:self.assertLessEqual(sum(instant-15<t<=instant for t in starts),4)
            self.assertGreaterEqual(max(starts),30)

    def test_eval_artifacts_and_optional_human_calibration(self):
        import numpy as np
        import pandas as pd
        for name,index in [(FILES[2],7),(FILES[3],10)]:
            ns=judge_namespace(name);nb=notebook(name);definitions(nb,index,ns)
            def hard(text,spec):
                passed=text.endswith('full')
                return {'partial_score':1. if passed else .5,'all_hard_passed':passed,'details':[{'id':'hard','passed':passed}]}
            def detail(text,prompt,soft,provider=None):
                provider=provider or 'gemma';score=int(text.endswith('full')) if provider=='gemma' else 0
                criteria=ns['parse_spec'](soft,'soft')
                return {'score':float(score),'scores':[{'id':s['id'],'category':s['category'],'score':score,'reason':'Mock evidence.'} for s in criteria],
                        'provider':provider,'model':'mock-'+provider,'base_url':'mock://'+provider,'rubric_version':'soft-v2'}
            model=SimpleNamespace(eval=lambda:None,gradient_checkpointing_disable=lambda:None,config=SimpleNamespace(use_cache=False))
            values=[json.loads(l) for l in (ROOT/DATA[1]).read_text().splitlines()[:2]]
            rows=[ns['normalize_row'](r) for r in values]
            baseline={ns['baseline_key']('seen',r):{'text':'before partial','generated_tokens':4,'truncated':False} for r in rows}
            ns.update(np=np,pd=pd,tqdm=lambda iterable,**kw:iterable,display=lambda *a:None,
                      eval_model=model,trainer=SimpleNamespace(model=model),EVAL_ADAPTER_DIR='mock',ADAPTER_DIR='mock',ADAPTER_STEP=1,
                      validation_callback=SimpleNamespace(best_step=1),EVAL_JUDGE_MODEL='mock-gemma',JUDGE_BASE_URL='mock://gemma',
                      generate_sample_response=lambda model,tok,prompt,**kw:{'text':'after full','generated_tokens':5,'truncated':False},
                      tokenizer=object(),hard_result=hard,evaluate_single_soft_details=detail,EVALUATE_BENCHMARK_BASELINE=True,
                      baseline_outputs=baseline,TRAIN_PROMPTS=set(),overlaps_training_prompt=lambda prompt:False)
            with tempfile.TemporaryDirectory() as d:
                ns.update(RESULTS_DIR=d,EVAL_RESULTS_DIR=d)
                with redirect_stdout(StringIO()):df,metrics=ns['run_benchmark']('seen',rows)
                self.assertEqual(metrics['soft_given_all_hard_after'],1.)
                self.assertEqual(metrics['joint_all_pass_after'],1.)
                self.assertEqual(metrics['joint_all_pass_before'],0.)
                outputs=json.loads((Path(d)/'eval_outputs_seen.json').read_text())
                self.assertTrue(outputs[0]['soft_judge_after']['scores'][0]['reason'])
                per_criterion=pd.read_csv(Path(d)/'eval_soft_criteria_seen.csv')
                self.assertEqual(len(per_criterion),8)
                self.assertTrue((Path(d)/'eval_soft_categories_seen.csv').is_file())
                calibration_index=9 if 'colab' in name else 12
                definitions(nb,calibration_index,ns)
                ns.update(csv=csv,random=random,Counter=Counter)
                with redirect_stdout(StringIO()):report=ns['run_judge_calibration'](d,max_responses=4,providers=['gemma','gemini'])
                self.assertEqual(report['human_calibration_status'],'pending')
                self.assertIsNone(report['agreement_with_human']['gemma']['overall']['agreement'])
                template=Path(d)/'judge_calibration/human_labels_template.csv'
                with template.open(encoding='utf-8-sig',newline='') as f:labels=list(csv.DictReader(f))
                self.assertNotIn('judge_score',labels[0])
                self.assertTrue(all(row['score']=='' for row in labels))
                by_response={row['label_id']:int(row['response'].endswith('full')) for row in labels}
                for row in labels:row['score']=str(by_response[row['label_id']])
                human=Path(d)/'human_labels.csv'
                with human.open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(labels[0]));writer.writeheader();writer.writerows(labels)
                with redirect_stdout(StringIO()):report=ns['run_judge_calibration'](d,str(human),4,['gemma','gemini'])
                self.assertEqual(report['human_calibration_status'],'measured')
                self.assertEqual(report['agreement_with_human']['gemma']['overall']['agreement'],1.)
                self.assertEqual(report['agreement_with_human']['gemma']['overall']['cohen_kappa'],1.)
                self.assertLess(report['agreement_with_human']['gemini']['overall']['agreement'],1.)
                self.assertEqual(ns['binary_agreement']([1,1],[1,1])['cohen_kappa'],None)
                with self.assertRaises(ValueError):ns['run_judge_calibration'](d,str(template),4,['gemma'])


if __name__=='__main__':
    unittest.main(verbosity=2)
