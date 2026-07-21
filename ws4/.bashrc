mkws () {
    if [ -z "$1" ]; then
        echo "Usage: mkws <worksheet-name>"
        return 1
    fi
    mkdir -p ~/cs131/"$1"
    cd ~/cs131/"$1"
    touch answer.txt
    vim answer.txt
}

alias gpush='git add -A && git commit -m "update worksheet" && git push'
if [[ "$PWD" == "$HOME/cs131"* ]]; then
    echo "You're in your cs131 repo — gpush and mkws are ready to use."
fi
