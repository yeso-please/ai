"""Read-only serving check of the saved E5 checkpoint; no training or DB access."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('data/interim/models/e5-base-rating-2022-2023-t4-all-v2/final'))
    p.add_argument('--output', type=Path, default=Path('reports/e5_serving_check.json'))
    args = p.parse_args()
    info = json.loads((args.model_dir/'training_info.json').read_text(encoding='utf-8'))
    if info['input_format'] != 'e5' or info['embedding_dimension'] != 768 or info['max_seq_length'] != 256:
        raise ValueError('Unexpected checkpoint contract')
    os.environ.update(MODEL_NAME=str(args.model_dir.resolve()), MODEL_VERSION=info['version'],
                      MODEL_INPUT_FORMAT='e5', TEMPLATE_VERSIONS='3', MAX_BATCH_ITEMS='8')
    from fastapi.testclient import TestClient
    import app.main as serving
    from tripin_ai.encoder import from_base64
    from tripin_ai.templates import TravelerSurvey, traveler_text
    start = time.perf_counter()
    with TestClient(serving.app) as client:
        deadline = time.monotonic()+180
        while client.get('/health').json()['status'] == 'LOADING' and time.monotonic() < deadline:
            time.sleep(.1)
        health = client.get('/health').json()
        assert health['status'] == 'UP' and health['dimension'] == 768 and health['templateVersions'] == [3], health
        styles, motives = {1:1,3:4,5:6,6:7}, [2,7]
        request = dict(requestId='synthetic-serving-check',modelVersion=info['version'],templateVersion=3,
                       profile=dict(travelStyles=styles,travelMotives=motives,likedRegions=['무시할 지역']))
        profile = client.post('/embeddings',json=request)
        assert profile.status_code == 200, profile.status_code
        user_vector = from_base64(profile.json()['embeddingBase64'])
        # Direct call uses exactly the prefix/text/length of the training code.
        expected = serving.state.encoder.model.encode(
            ['query: '+traveler_text(TravelerSurvey(styles=styles,motives=motives))],
            normalize_embeddings=True,convert_to_numpy=True)[0]
        np.testing.assert_allclose(user_vector,expected,rtol=1e-5,atol=1e-6)
        item = dict(id='synthetic-palace',name='경복궁',contentTypeId=12,lclsSystm1='HS',lclsSystm2='HS01',
                    lclsSystm3='HS010100',regionName='서울특별시 종로구',description='조선의 궁궐')
        attractions = client.post('/embeddings/batch',json=dict(modelVersion=info['version'],templateVersion=3,items=[item]))
        assert attractions.status_code == 200, attractions.status_code
        item_vector = from_base64(attractions.json()['items'][0]['embeddingBase64'])
        from app.schemas import AttractionIn
        expected_item = serving.state.encoder.model.encode(
            ['passage: '+serving.compose_attraction(AttractionIn(**item),3)],
            normalize_embeddings=True,convert_to_numpy=True)[0]
        np.testing.assert_allclose(item_vector,expected_item,rtol=1e-5,atol=1e-6)
        for vector in [user_vector,item_vector]:
            assert vector.shape == (768,) and np.isfinite(vector).all()
            assert abs(float(np.linalg.norm(vector))-1) < 1e-5
        assert client.post('/embeddings',json={**request,'modelVersion':'old-model'}).status_code == 409
        assert client.post('/embeddings',json={**request,'templateVersion':2}).status_code == 409
    result = dict(status='PASS',health=health,training_input_parity=True,query_and_passage_prefixes=True,
                  dimensions=768,bytes_per_vector=3072,l2_normalized=True,old_versions_rejected=True,
                  elapsed_seconds=round(time.perf_counter()-start,2),database_access=False,training_performed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
