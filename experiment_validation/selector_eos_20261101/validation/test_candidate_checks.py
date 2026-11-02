"""Fault-injection checks using the retained preflight evidence, without inference."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from transformers import AutoTokenizer
from candidate_checks import validate_call, FIRST

HERE=Path(__file__).resolve().parent


class CandidateEvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run=HERE.parent/'preflight/main_smoke'
        protocol=json.loads((run/'protocol.json').read_text())
        record=json.loads((run/'samples.jsonl').read_text().splitlines()[0])
        cls.call=record['actual_encoder_inputs'][0]
        cls.tokenizer=AutoTokenizer.from_pretrained(FIRST,revision=protocol['checkpoints'][FIRST],local_files_only=True,use_fast=True)

    def test_untouched_saved_evidence(self):
        counts,_=validate_call(self.call,self.tokenizer,3)
        self.assertEqual(counts['candidates'],3)

    def test_included_score_tampering_detected(self):
        c=deepcopy(self.call);c['candidate_audit']['candidates'][0]['transition_scores'][0]-=0.5
        with self.assertRaisesRegex(ValueError,'sum mismatch'):
            validate_call(c,self.tokenizer,3)

    def test_eos_offset_tampering_detected(self):
        c=deepcopy(self.call);a=c['candidate_audit']['candidates'][0]
        a['first_eos_generated_offset']=-1
        with self.assertRaisesRegex(ValueError,'EOS offset'):
            validate_call(c,self.tokenizer,3)

    def test_selected_candidate_tampering_detected(self):
        c=deepcopy(self.call);c['candidate_audit']['selected_candidate_index']=99
        with self.assertRaisesRegex(ValueError,'selected candidate'):
            validate_call(c,self.tokenizer,3)

    def test_token_text_tampering_detected(self):
        c=deepcopy(self.call);c['candidate_audit']['candidates'][0]['text']+=' fabricated'
        with self.assertRaisesRegex(ValueError,'text/eligibility'):
            validate_call(c,self.tokenizer,3)

    def test_single_candidate_bypass_contract(self):
        c=deepcopy(self.call);a=c['candidate_audit'];candidate=a['candidates'][0]
        a.update(samples_requested=1,score_dtype=None,candidates=[candidate],ranked_candidate_indices=[0],
                 ranked_texts=[candidate['text']],selected_candidate_index=0)
        candidate.update(confidence=0.,transition_scores=None,scored_token_count=0,legacy_full_step_confidence=None)
        c.update(samples=1,selected_text=candidate['text'])
        counts,_=validate_call(c,self.tokenizer,1)
        self.assertEqual(counts['single_candidate_calls'],1)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CandidateEvidenceChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'scope':'Saved multi-candidate evidence, corruption rejection and singleton no-score contract; no new inference.'}
    (HERE/'validation_selftest.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
