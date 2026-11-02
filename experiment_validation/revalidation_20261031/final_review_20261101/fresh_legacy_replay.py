"""CPU-only fixed-ID replay of archived single-token detokenization.

Fresh IDs are known; historical IDs are not. Exact historical text/vector matches
demonstrate output consistency, not historical checkpoint or token-ID recovery.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def htext(text):return hashlib.sha256(text.encode()).hexdigest()


def readrows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    before=BASE/'references/app_before.py'
    source=before.read_text()
    assert 'def _decode_single_token(tok_id: int)' in source
    assert 'skip_special_tokens=False' in source and 'clean_up_tokenization_spaces=False' in source
    report={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
            'legacy_source':str(before.relative_to(ROOT)),'legacy_source_sha256':sha(before),
            'legacy_call':'tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)',
            'method':'Use the recorded fresh sequence of token IDs with its pinned victim tokenizer. Compare concatenated independent-token decoding with full-sequence decoding and both sets of per-ID fragments/UTF-8 lengths against archived logs.',
            'scope':'CPU tokenization only; no victim or reconstruction inference; no attested artifact edits.',
            'models':[],'limits':['Historical logs do not contain token IDs; exact decoded text/fragments/length matches do not recover or prove original IDs.',
                'Current pinned tokenizer/model revisions are not independent attestations of historical checkpoint versions.',
                'Within each recorded fresh ID sequence, switching the decoding function isolates its effect on emitted strings and byte vectors.',
                'This replay does not rerun T5 on the legacy replay and does not by itself isolate the causal contribution of detokenization to changes in reconstruction scores.']}
    allrows=[]
    for dirname in ['tinyllama','phi']:
        folder=BASE/'fresh_references'/dirname
        rows=readrows(folder/'canonical_traces.jsonl');assert len(rows)==300 and len({r['idx'] for r in rows})==300
        protocol=json.loads((folder/'protocol.json').read_text())
        model_id,revision=protocol['model_id'],protocol['revision']
        cache=Path.home()/'.cache/huggingface/hub'/('models--'+model_id.replace('/','--'))/'snapshots'/revision
        for filename in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
            assert sha(cache/filename)==protocol['checkpoint_file_sha256'][filename]
        tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True,use_fast=True)
        decoded={};counts=Counter();different_ids=Counter();examples=[]
        for row in rows:
            freshlogpath=ROOT/row['generation_log_path'];assert sha(freshlogpath)==row['generation_log_sha256']
            freshlog=json.loads(freshlogpath.read_text())
            oldlog=json.loads((ROOT/'logs'/(row['historical_run_id']+'.json')).read_text())
            assert htext(oldlog['response_text'])==row['historical_reference_sha256']
            assert oldlog['prompt']==freshlog['prompt']==row['prompt']
            ids=row['generated_token_ids'];assert ids==freshlog['generated_token_ids']
            full=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert full==row['reference']==freshlog['response_text']
            for tok_id in set(ids):
                if tok_id not in decoded:
                    decoded[tok_id]=tokenizer.decode([tok_id],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            legacy_fragments=[decoded[t] for t in ids]
            contextual_fragments=[s['token_text'] for s in freshlog['steps']]
            old_fragments=[s['token_text'] for s in oldlog['steps']]
            assert ids==[s['token_id'] for s in freshlog['steps']]
            legacy=''.join(legacy_fragments)
            legacy_lengths=[len(s.encode()) for s in legacy_fragments]
            contextual_lengths=[len(s.encode()) for s in contextual_fragments]
            historical_lengths=[s['token_utf8_len'] for s in oldlog['steps']]
            assert contextual_lengths==row['token_lengths'] and ''.join(contextual_fragments)==full
            exact={
                'legacy_text_equals_historical':legacy==oldlog['response_text'],
                'legacy_fragments_equal_historical':legacy_fragments==old_fragments,
                'legacy_lengths_equal_historical':legacy_lengths==historical_lengths,
                'fresh_text_equals_historical':full==oldlog['response_text'],
                'fresh_fragments_equal_historical':contextual_fragments==old_fragments,
                'fresh_lengths_equal_historical':contextual_lengths==historical_lengths,
                'legacy_text_equals_full_sequence':legacy==full,
                'legacy_and_full_equal_after_removing_ascii_spaces':legacy.replace(' ','')==full.replace(' ',''),
                'legacy_lengths_equal_contextual':legacy_lengths==contextual_lengths,
                'legacy_without_ascii_space':' ' not in legacy,
                'fresh_without_ascii_space':' ' not in full,
                'historical_without_ascii_space':' ' not in oldlog['response_text'],
                'legacy_has_replacement_character':'\ufffd' in legacy,
                'fresh_has_replacement_character':'\ufffd' in full,
                'same_token_count_as_historical':len(ids)==len(old_fragments),
            }
            for name,value in exact.items():counts[name]+=value
            changed_positions=sum(a!=b for a,b in zip(legacy_fragments,contextual_fragments))
            counts['total_fresh_tokens']+=len(ids)
            counts['changed_fragment_positions_same_fresh_ids']+=changed_positions
            counts['changed_fragments_are_one_restored_leading_ascii_space']+=sum(b==' '+a for a,b in zip(legacy_fragments,contextual_fragments) if a!=b)
            counts['legacy_ascii_spaces']+=legacy.count(' ')
            counts['fresh_ascii_spaces']+=full.count(' ')
            counts['historical_ascii_spaces']+=oldlog['response_text'].count(' ')
            counts['legacy_payload_bytes']+=sum(legacy_lengths)
            counts['fresh_payload_bytes']+=sum(contextual_lengths)
            counts['historical_payload_bytes']+=sum(historical_lengths)
            for tok_id,a,b in zip(ids,legacy_fragments,contextual_fragments):
                if a!=b:different_ids[tok_id]+=1
            item={'label':row['label'],'idx':row['idx'],'fresh_run_id':row['run_id'],'historical_run_id':row['historical_run_id'],
                  'fresh_token_count':len(ids),'historical_step_count':len(old_fragments),**exact,
                  'legacy_text_sha256':htext(legacy),'fresh_text_sha256':htext(full),
                  'historical_text_sha256':row['historical_reference_sha256'],
                  'legacy_text':legacy,'legacy_token_lengths':legacy_lengths,
                  'ascii_spaces':{'legacy':legacy.count(' '),'fresh':full.count(' '),'historical':oldlog['response_text'].count(' ')},
                  'changed_fragment_positions':changed_positions}
            allrows.append(item)
            if len(examples)<3 and not exact['legacy_text_equals_full_sequence']:
                examples.append({'idx':row['idx'],'legacy_prefix':legacy[:180],'full_sequence_prefix':full[:180],
                                 'legacy_equals_historical':exact['legacy_text_equals_historical']})
        report['models'].append({'label':rows[0]['label'],'model_id':model_id,'tokenizer_revision':revision,'n':300,
             'source_inventory_sha256':sha(folder/'canonical_traces.jsonl'),
             'tokenizer_json_sha256':sha(cache/'tokenizer.json'),'counts':dict(counts),
             'unique_token_ids_replayed':len(decoded),'unique_token_ids_with_changed_fragments':len(different_ids),
             'illustrative_examples_not_prevalence_estimates':examples})
    (HERE/'fresh_legacy_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (HERE/'fresh_legacy_replay_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
    lines=['# Stałe sekwencje ID: porównanie dawnej i pełnej detokenizacji','',
           'Rekonstrukcja leksykalna nie była tutaj ponownie uruchamiana. Użyto dokładnie tych samych, zapisanych ID z nowych generacji i przypiętych tokenizerów. Dawna funkcja dekodowała każdy token oddzielnie z `skip_special_tokens=False` i `clean_up_tokenization_spaces=False`.', '',
           '| Model | n | Dawny replay = historyczny tekst | Dawny replay = historyczne fragmenty | Dawny replay = historyczne długości | Dawny replay = pełna nowa detokenizacja | Bez spacji: dawny replay / nowy tekst / archiwum |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in report['models']:
        c=m['counts']
        lines.append(f"| {m['label']} | 300 | {c['legacy_text_equals_historical']} | {c['legacy_fragments_equal_historical']} | {c['legacy_lengths_equal_historical']} | {c['legacy_text_equals_full_sequence']} | {c['legacy_without_ascii_space']} / {c['fresh_without_ascii_space']} / {c['historical_without_ascii_space']} |")
    lines+=['','W obrębie zapisanej nowej sekwencji ID kontrast izoluje wpływ funkcji dekodującej na tekst i wektor bajtów. Zgodność z archiwalnym tekstem, fragmentami lub długościami nie odzyskuje nieobecnych historycznych ID i nie poświadcza historycznej wersji checkpointu. Różnicy wyników T5 między kampaniami nie przypisuje się na tej podstawie wyłącznie detokenizacji.','']
    (HERE/'fresh_legacy_replay.md').write_text('\n'.join(lines))
    print(json.dumps({'status':report['status'],'models':[{'label':m['label'],'n':m['n'],'counts':m['counts']} for m in report['models']]},indent=2))


if __name__=='__main__':main()
