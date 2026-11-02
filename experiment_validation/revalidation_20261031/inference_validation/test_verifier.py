"""Integrity rejection checks using the existing real CPU reconstruction."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

PATH = Path(__file__).with_name('verify_inference.py')
SPEC = importlib.util.spec_from_file_location('inference_verifier_under_test', PATH)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class ExistingRecordChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = v.BASE/'inference/cpu_verified_smoke'
        protocol = json.loads((run/'protocol.json').read_text())
        cls.row = json.loads((run/'samples.jsonl').read_text().splitlines()[0])
        cls.source = json.loads((v.BASE/'collection/canonical_traces.jsonl').read_text().splitlines()[0])
        cls.source['_has_old_comparison'] = True
        cls.source['_scenario'] = v.scenario(cls.source)
        cls.tokenizers = {}
        cache = Path.home()/'.cache/huggingface/hub'
        for model, revision in protocol['checkpoints'].items():
            path = cache/('models--'+model.replace('/','--'))/'snapshots'/revision/'tokenizer.json'
            raw, limited = Tokenizer.from_file(str(path)), Tokenizer.from_file(str(path))
            raw.no_padding(); raw.no_truncation(); limited.no_padding(); limited.enable_truncation(max_length=80)
            cls.tokenizers[model] = (raw,limited)

    def verify(self, row):
        return v.validate_record(row,self.source,self.row['protocol_id'],self.tokenizers)

    def test_real_record_and_full_segment_coverage(self):
        result = self.verify(self.row)
        self.assertEqual(result['attempted_segments'],35)
        self.assertEqual(result['available_segments'],35)
        self.assertEqual(result['unattempted_segments'],0)
        self.assertAlmostEqual(result['metrics']['phi'],self.row['phi'],places=12)

    def test_corruption_is_rejected(self):
        mutations = {
            'reference': lambda r:r.__setitem__('reference','changed'),
            'metric': lambda r:r.__setitem__('phi',.99),
            'missing_segment': lambda r:r['actual_encoder_inputs'].pop(),
            'encoder_token': lambda r:r['actual_encoder_inputs'][0]['retained_token_ids'].__setitem__(0,99),
            'context': lambda r:r['actual_encoder_inputs'][1].__setitem__('text',r['actual_encoder_inputs'][1]['text'].replace(r['first_prediction'],'changed')),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                row = copy.deepcopy(self.row); mutate(row)
                with self.assertRaises(ValueError):
                    self.verify(row)

    def test_segmentation_matches_published_helper_on_random_traces(self):
        import weiss_reconstruction as wr
        rng = random.Random(726)
        for _ in range(1000):
            values = [rng.randrange(14) for _ in range(rng.randrange(100))]
            self.assertEqual(v.independent_segments(values),wr.heuristic_sentences_from_lengths(values))

    def test_fresh_reference_does_not_treat_placeholder_as_historical(self):
        source = copy.deepcopy(self.source)
        source['_has_old_comparison'] = False
        source['old_pred_full_text'] = ''
        row = copy.deepcopy(self.row)
        row['old_phi'] = 0.0
        result = v.validate_record(row,source,self.row['protocol_id'],self.tokenizers)
        self.assertIsNone(result['old_metrics'])
        self.assertIsNone(result['delta_phi'])


class FreshDecodeChecks(unittest.TestCase):
    def test_full_decode_not_just_step_concatenation(self):
        tokenizer = Tokenizer(WordLevel({'[UNK]':0,'hello':1,'world':2},unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        protocol = {'model_id':'example/pinned','harness_source_sha256':'same'}
        pid = v.digest(json.dumps(protocol,sort_keys=True).encode())
        run_dir = Path('/unused/test-only-cache-key')
        steps = [dict(token_id=1,token_text='hello',step=0,sse_frame_utf8_len=13),
                 dict(token_id=2,token_text=' world',step=1,sse_frame_utf8_len=14)]
        wire = b'data: hello\n\ndata:  world\n\ndata: [DONE]\n\n'
        row = dict(source_kind='fresh_contextual_reference',historical_comparison_available=False,
                   reference_validation='contextual-full-sequence-decoding-verified-v1; fresh pinned victim generation',
                   run_dir=str(run_dir),protocol_id=pid,model_id='example/pinned',generated_token_ids=[1,2],
                   reference='hello world',token_lengths=[5,6],input_text='hello',input_token_ids=[1],
                   emitted_stream_bytes=len(wire),emitted_stream_sha256=v.digest(wire))
        log = dict(model_id='example/pinned',harness_source_sha256='same',
                   detokenization_policy='contextual_prefix_utf8_buffer_v1',generated_token_ids=[1,2],
                   generated_tokens=2,steps=steps)
        cached = {run_dir:(protocol,tokenizer)}
        v.validate_fresh_reference(row,log,cached)
        # Preserve all text/frame consistency while changing actual IDs.
        badrow,badlog = copy.deepcopy(row),copy.deepcopy(log)
        badrow['generated_token_ids']=[2,1]
        badlog['generated_token_ids']=[2,1]
        badlog['steps'][0]['token_id']=2
        badlog['steps'][1]['token_id']=1
        with self.assertRaisesRegex(ValueError,'full-sequence tokenizer decoding'):
            v.validate_fresh_reference(badrow,badlog,cached)


if __name__ == '__main__':
    unittest.main()
