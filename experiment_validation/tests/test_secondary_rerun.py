"""Integrity checks for a resumable secondary inference campaign; no models."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_validation.revalidation_20261031.secondary import rerun_secondary as run
from experiment_validation.scripts import run_validation as reference


class SecondaryRerunTests(unittest.TestCase):
    def item(self):
        source=dict(idx=1,topic='topic',prompt='prompt',run_id='run',run_dir='directory',model='model',
            reference='alpha beta',reference_sha256='reference-digest',generation_log_sha256='log-digest',
            clean_lengths=[2,3,4],clean_lengths_sha256=run.digest([2,3,4]),victim_budget=96,
            source_response_complete=None,reference_scope='emitted text')
        return run.make_item(source,'qwen_1_5b','trace_control','real','frozen',20260706,123,None,[2,3,4],[2,3,4])

    def record(self,item=None,protocol='protocol'):
        item=self.item() if item is None else item
        n=len(run.wr.heuristic_sentences_from_lengths(item['attacker_input_lengths']))
        text=run.wr.make_input_from_lengths(run.wr.heuristic_sentences_from_lengths(item['attacker_input_lengths'])[0])
        tokenizer=run.pinned_tokenizer(run.wr.FIRST_SENTENCES_MODEL)
        full=tokenizer([text],padding=True,truncation=False)['input_ids'][0]
        kept=tokenizer([text],padding=True,truncation=True,max_length=80)['input_ids'][0]
        inputs=[dict(text=text,full_token_ids=full,full_token_count=len(full),retained_token_ids=kept,
                     truncated=len(full)>len(kept),max_length=80,samples=3,model_id=run.wr.FIRST_SENTENCES_MODEL,selected_text='alpha gamma')]
        result=SimpleNamespace(full_text='alpha gamma',first_sentence='alpha gamma',sentence_count=1,
                               available_segment_count=n)
        return run.record_from_result(item,result,inputs,protocol,.1)

    def test_valid_resume_and_protocol_or_input_tampering_rejected(self):
        item=self.item();record=self.record(item)
        with TemporaryDirectory() as td:
            p=Path(td)/'samples.jsonl';p.write_text(run.packed(record)+'\n')
            self.assertEqual(len(run.load_completed(p,[item],'protocol')),1)
            with self.assertRaises(ValueError):run.load_completed(p,[item],'another-protocol')
            changed=deepcopy(item);changed['reference']='different reference'
            with self.assertRaises(ValueError):run.load_completed(p,[changed],'protocol')
            changed=deepcopy(record);changed['metrics']['phi']=0.99
            changed['record_sha256']=run.digest({k:v for k,v in changed.items() if k!='record_sha256'})
            p.write_text(run.packed(changed)+'\n')
            with self.assertRaises(ValueError):run.load_completed(p,[item],'protocol')

    def test_duplicate_and_torn_rows_fail_closed(self):
        record=self.record();item=self.item()
        with TemporaryDirectory() as td:
            p=Path(td)/'samples.jsonl';p.write_text((run.packed(record)+'\n')*2)
            with self.assertRaises(ValueError):run.load_completed(p,[item],'protocol')
            p.write_text(run.packed(record)+'\n{"task_id":')
            with self.assertRaises(ValueError):run.load_completed(p,[item],'protocol')

    def test_invalid_segment_provenance_rejected_even_with_fresh_hash(self):
        item=self.item();record=self.record(item);record['attempted_segments']=0;record['actual_encoder_inputs']=[]
        record['record_sha256']=run.digest({k:v for k,v in record.items() if k!='record_sha256'})
        with self.assertRaises(ValueError):run.validate_record(record,item,'protocol')

    def test_encoder_ids_and_selected_output_checked_independently(self):
        item=self.item()
        for field,value in [('retained_token_ids',[42]),('selected_text','different output'),('text','incorrect serialization')]:
            record=self.record(item);record['actual_encoder_inputs'][0][field]=value
            record['record_sha256']=run.digest({k:v for k,v in record.items() if k!='record_sha256'})
            with self.assertRaises(ValueError):run.validate_record(record,item,'protocol')

    def test_short_run_requires_empty_final_middle_output(self):
        original=self.item();lengths=[2]*20+[1]+[3]*20+[1]+[4]*20
        source={k:original[k] for k in ['idx','topic','prompt','run_id','run_dir','model','reference',
            'reference_sha256','generation_log_sha256','victim_budget','source_response_complete','reference_scope']}
        source.update(clean_lengths=lengths,clean_lengths_sha256=run.digest(lengths))
        item=run.make_item(source,'qwen_1_5b','trace_control','real','frozen',20260706,123,None,lengths,lengths)
        segments=run.wr.heuristic_sentences_from_lengths(lengths);self.assertEqual(len(segments),3)
        actual=[]
        for i,selected in enumerate(['alpha gamma','']):
            model_id=run.wr.FIRST_SENTENCES_MODEL if i==0 else run.wr.MIDDLE_SENTENCES_MODEL
            text=run.wr.make_input_from_lengths(segments[i]) if i==0 else run.wr.make_input_with_context(segments[i],'alpha gamma')
            tokenizer=run.pinned_tokenizer(model_id)
            full=tokenizer([text],padding=True,truncation=False)['input_ids'][0]
            kept=tokenizer([text],padding=True,truncation=True,max_length=80)['input_ids'][0]
            actual.append(dict(model_id=model_id,text=text,full_token_ids=full,full_token_count=len(full),
                retained_token_ids=kept,truncated=len(full)>len(kept),samples=3,max_length=80,selected_text=selected))
        result=SimpleNamespace(full_text='alpha gamma',first_sentence='alpha gamma',sentence_count=1,available_segment_count=3)
        record=run.record_from_result(item,result,actual,'protocol',.1)
        run.validate_record(record,item,'protocol')
        record['actual_encoder_inputs'][-1]['selected_text']='continued'
        record['prediction']='alpha gamma continued';record['metrics']=run.metrics(item['reference'],record['prediction'])
        record['reconstructed_segments']=2
        record['record_sha256']=run.digest({k:v for k,v in record.items() if k!='record_sha256'})
        with self.assertRaisesRegex(ValueError,'Missing decoder calls'):run.validate_record(record,item,'protocol')

    def test_metrics_match_existing_independent_dp_for_unicode_and_empty(self):
        examples=[('', ''),('', 'alpha'),('alpha',''),('alpha beta','alpha gamma'),('żółć\né 🐈','żółć é'),
                  ('a b c a b','a a c b')]
        for a,b in examples:
            got=run.metrics(a,b)
            self.assertEqual(got['ed_norm'],reference.levenshtein_distance(a,b)/max(1,len(a)))
            for name,fn in [('rouge1',reference.rouge1_scores),('rougeL',reference.rouge_l_scores)]:
                p,r,f=fn(a,b)
                self.assertEqual((got[name+'_precision'],got[name+'_recall'],got[name+'_f1']),(p,r,f))

    def test_partial_design_cannot_be_complete_or_generate_contrasts(self):
        item=self.item();second=deepcopy(item);second['task_id']='missing';second['idx']=2
        inventory=[item,second]
        for suite in ['defense_pilot','map']:
            other=deepcopy(item);other.update(task_id=suite,suite=suite,condition='baseline');inventory.append(other)
        got=run.aggregate([self.record(item)],inventory,'protocol')
        self.assertFalse(got['complete']);self.assertEqual(got['paired_contrasts'],[])
        self.assertEqual(got['suite_complete'],dict(trace_control=False,defense_pilot=False,map=False))
        self.assertNotIn('phi_ci_prompt',got['cells'][0])

    def test_actual_3900_inventory_and_map_common_seeds(self):
        inventory,calibration,sources,audit,config=run.build_inventory()
        self.assertEqual(len(inventory),3900);self.assertEqual(audit['validated_generation_logs'],660)
        self.assertEqual(audit['map_archived_keys_seeds_counts_matched'],3300)
        self.assertEqual({k:len(v) for k,v in calibration.items()},{'qwen_1_5b':270,'llama_3_2_3b':270})
        common={}
        for item in inventory:
            if item['suite']=='map':
                key=item['label'],item['idx'],item['seed_index']
                common.setdefault(key,item['decoder_seed']);self.assertEqual(common[key],item['decoder_seed'])
            if item['condition']=='cross_prompt':
                self.assertEqual(item['donor']['idx'],item['idx']%30+1)
                self.assertNotEqual(item['run_id'],item['donor']['run_id'])
            if item['condition']=='rand_pad_8':
                increments=[a-b for a,b in zip(item['observed_lengths'],item['clean_lengths'])]
                self.assertTrue(all(0<=n<=8 for n in increments))
                self.assertAlmostEqual(item['observed_payload_overhead_pct'],100*sum(increments)/sum(item['clean_lengths']))


if __name__=='__main__':unittest.main()
