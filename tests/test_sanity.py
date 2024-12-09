import krx_eval


def test_sanity():
    assert krx_eval

def test_vllm():
    import vllm
    assert vllm

def test_nltk():
    import nltk
    assert nltk
    
def test_litellm():
    import litellm
    assert litellm

def test_konlpy():
    import konlpy
    assert konlpy
