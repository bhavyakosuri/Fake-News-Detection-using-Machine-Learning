import os

def test_repo_layout():
    assert os.path.exists('train.py')
    assert os.path.exists('app.py')
    assert os.path.exists('templates/index.html')
