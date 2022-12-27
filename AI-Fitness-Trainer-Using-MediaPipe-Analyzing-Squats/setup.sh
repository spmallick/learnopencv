mkdir -p ~/.streamlit/
echo "
[server]\n
headless=true\n
enableCORS=false\n
enableXsrfProtection=false\n
port=8080\n
\n
" > ~/.streamlit/config.toml