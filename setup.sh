mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"irsandiagep@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $8080\n\
" > ~/.streamlit/config.toml