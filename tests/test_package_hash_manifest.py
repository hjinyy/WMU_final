from pathlib import Path
import csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
def test_package_sha256_inventory_matches_files():
 p=ROOT/'data/package_sha256.csv'
 rows=list(csv.DictReader(p.open(newline='',encoding='utf-8')))
 assert len(rows)>=70
 for row in rows:
  f=ROOT/row['Path'];assert f.is_file(),row['Path']
  assert f.stat().st_size==int(row['Bytes']),row['Path']
  assert hashlib.sha256(f.read_bytes()).hexdigest()==row['SHA256'],row['Path']
