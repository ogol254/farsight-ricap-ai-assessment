"""Non-sensitive end-to-end deployment verification; creates labelled demo records.

Usage: python tests/verify_live.py https://your-host.example
No infrastructure secret is needed. Never prints the short-lived session token.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import time

import httpx
from PIL import Image, ImageDraw, ImageFont


def verify(base):
    checks = []
    with httpx.Client(base_url=base.rstrip('/'), timeout=180, follow_redirects=True) as client:
        def check(name, condition, detail=None):
            assert condition, f'{name}: {detail}'
            checks.append({'check': name, 'passed': True})
            print('PASS', name, flush=True)

        r = client.get('/ready')
        check('readiness', r.status_code == 200, r.text[:300])
        ready = r.json()
        check('PostgreSQL and private photo storage', ready.get('database') == 'postgresql' and ready.get('photo_storage') == 'private_supabase', ready)
        check('models, OCR and local LLM loaded', all(ready.get(k) for k in ['model_loaded', 'ocr_loaded', 'classifier_loaded', 'llm_loaded']), ready)
        check('UI delivered', 'Four services' in client.get('/').text)
        check('OpenAPI delivered', client.get('/openapi.json').status_code == 200)
        check('missing token rejected', client.get('/v1/history').status_code == 401)
        client.headers['Authorization'] = 'Bearer ' + client.post('/v1/sessions').json()['access_token']
        r = client.get('/v1/taxpayers')
        check('persisted revenue feature pipeline', r.status_code == 200 and len(r.json().get('records', [])) == 20, r.text[:300])
        r = client.post('/v1/audit-batches')
        check('UC1 DB-to-model audit batch saved', r.status_code == 200 and r.json().get('selected_count') == 20 and r.json().get('record_id'), r.text[:300])
        for lang, question in [('en', 'What payment channels are supported?'), ('so', 'Sidee lacagta loo bixin karaa?')]:
            r = client.post('/v1/assistant', json={'question': question})
            check('UC2 grounded ' + lang, r.status_code == 200 and r.json().get('grounded') and r.json().get('citations') and r.json().get('language') == lang, r.text[:300])
        r = client.post('/v1/assistant', json={'question': 'Who won the football match?'})
        check('UC2 unrelated question abstains', r.status_code == 200 and not r.json().get('grounded'), r.text[:300])
        image = Image.new('RGB', (850, 250), 'white')
        font_path = next((p for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()), None)
        font = ImageFont.truetype(font_path, 110) if font_path else ImageFont.load_default(size=110)
        ImageDraw.Draw(image).text((55, 50), '001234', font=font, fill='black')
        data = BytesIO(); image.save(data, format='PNG')
        r = client.post('/v1/meter-readings', files={'image': ('verification-display.png', data.getvalue(), 'image/png')}, data={'previous_reading': '1200'})
        check('UC3 real image OCR and save', r.status_code == 200 and r.json().get('reading_value') == 1234 and r.json().get('record_id'), r.text[:500])
        record = r.json()
        photo_path = f"/v1/meter-readings/{record['record_id']}/photo"
        photo = client.get(photo_path)
        check('UC3 private photo retrieval', photo.status_code == 200 and photo.headers.get('content-type') == 'image/jpeg', photo.status_code)
        r = client.post(f"/v1/meter-readings/{record['record_id']}/review", json={'reading_value': 1234, 'reason': 'End-to-end test: checked synthetic display'})
        check('UC3 human review saved separately', r.status_code == 200 and r.json().get('original_ocr_preserved'), r.text[:300])
        r = client.post('/v1/meter-readings', files={'image': ('bad.jpg', b'not an image', 'image/jpeg')})
        check('UC3 invalid image rejected', r.status_code == 422, r.text[:300])
        r = client.post('/v1/complaints', json={'text': 'My payment is missing'})
        check('UC4 classification and routing saved', r.status_code == 200 and r.json().get('category') == 'payment' and r.json().get('record_id'), r.text[:300])
        records = client.get('/v1/history').json()['records']
        check('all workflow records retrievable', {'audit_batch', 'assistant', 'meter', 'meter_review', 'complaint'} <= {x['kind'] for x in records})
        client.headers['Authorization'] = 'Bearer ' + client.post('/v1/sessions').json()['access_token']
        check('second session cannot read first photo', client.get(photo_path).status_code == 404)
        check('second session has separate history', client.get('/v1/history').json()['records'] == [])
        check('still ready after inference', client.get('/ready').status_code == 200)
    return {'host': base, 'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'checks': checks,
            'limits': 'Synthetic display image, not a field meter or physical phone-camera test. No production accuracy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_url')
    args = parser.parse_args()
    print(json.dumps(verify(args.base_url), indent=2))
