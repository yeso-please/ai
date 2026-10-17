import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import app.main as main
from app.schemas import ProfileIn
from tripin_ai.encoder import Encoder
from tripin_ai.templates import TravelerSurvey, traveler_text


def fake_transformers(monkeypatch):
    class Model:
        max_seq_length = 512

        def __init__(self, *args, **kwargs):
            self.seen = []

        def get_sentence_embedding_dimension(self):
            return 768

        def encode(self, texts, **kwargs):
            self.seen.extend(texts)
            return np.ones((len(texts), 768), dtype=np.float32)/np.sqrt(768)

    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Model))


def test_e5_uses_training_metadata_prefixes_and_length(tmp_path, monkeypatch):
    fake_transformers(monkeypatch)
    (tmp_path/'training_info.json').write_text(json.dumps(dict(input_format='e5', max_seq_length=256,
                                                              embedding_dimension=768)), encoding='utf-8')
    encoder = Encoder(str(tmp_path), 'trained-e5')
    encoder.encode(['여행자'], role='query')
    result = encoder.encode(['관광지'], role='passage')
    assert encoder.model.seen == ['query: 여행자', 'passage: 관광지']
    assert encoder.model.max_seq_length == 256
    assert result.shape == (1,768) and result.dtype == np.dtype('<f4')
    with pytest.raises(ValueError, match='differs from training'):
        Encoder(str(tmp_path), 'trained-e5', input_format='plain')


def test_plain_encoder_keeps_legacy_text(monkeypatch):
    fake_transformers(monkeypatch)
    encoder = Encoder('fake-minilm')
    encoder.encode(['여행자'], role='query')
    assert encoder.model.seen == ['여행자']


def test_template_three_matches_training_and_does_not_change_two():
    survey = TravelerSurvey(styles={1:1,3:4,5:6,6:7}, motives=[2,7], liked_regions=['제주시'])
    expected = traveler_text(TravelerSurvey(styles=survey.styles, motives=survey.motives))
    assert traveler_text(survey, 'aihub-v2') == expected
    assert '제주시' in traveler_text(survey, 'aihub-v1')
    profile = ProfileIn(travel_styles=survey.styles, travel_motives=survey.motives,
                        liked_regions=['제주시'], travel_mbti='INFP', exclude_tags=['물놀이'])
    assert main.compose_profile(profile, 3) == expected


def test_api_uses_query_and_passage_roles(monkeypatch):
    from app.config import Settings
    from fastapi.testclient import TestClient
    seen = []

    class EncoderStub:
        dimension = 768

        def encode(self, texts, role='passage'):
            seen.append((role, texts))
            return np.ones((len(texts),768),dtype=np.float32)/np.sqrt(768)

    monkeypatch.setattr(main.state, 'settings', Settings('fake','e5-deploy',(3,),8,'e5'))
    monkeypatch.setattr(main.state, 'encoder', EncoderStub())
    monkeypatch.setattr(main, 'encoder_factory', lambda _: main.state.encoder)
    with TestClient(main.app) as client:
        result = client.post('/embeddings', json=dict(requestId='x',modelVersion='e5-deploy',templateVersion=3,
            profile=dict(travelStyles={'1':1,'3':4,'5':6,'6':7},travelMotives=[2],likedRegions=['제주시'])))
        assert result.status_code == 200 and result.json()['dimension'] == 768
        result = client.post('/embeddings/batch', json=dict(modelVersion='e5-deploy',templateVersion=3,
            items=[dict(id='1',name='경복궁',contentTypeId=12,lclsSystm1='HS',lclsSystm2='HS01',lclsSystm3='HS010100',
                        regionName='서울특별시 종로구',description='조선의 궁궐')]))
        assert result.status_code == 200 and result.json()['dimension'] == 768
        assert client.post('/embeddings', json=dict(requestId='x',modelVersion='old',templateVersion=3,
            profile=dict(travelStyles={'1':1}))).status_code == 409
    assert [role for role,_ in seen] == ['query','passage']
    assert '제주시' not in seen[0][1][0]
    assert '역사' in seen[1][1][0]
