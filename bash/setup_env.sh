```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install packaging ninja wheel
pip3 install -e ".[model_worker,webui]"
pip3 install flash-attn --no-build-isolation
pip3 install -e ".[train]"
pip3 install config openai tenacity gym nltk
```