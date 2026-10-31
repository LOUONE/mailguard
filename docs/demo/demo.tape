# The README's demo, recorded by VHS against the running stack: `make demo`, then `make demo-gif`.
# Both emails are made up (RFC 2606 domains, a documentation IP range).

Output demo.gif

Require curl
Require jq

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 17
Set Width 1100
Set Height 720
Set Padding 24
Set Theme "Catppuccin Mocha"
Set Framerate 20
Set TypingSpeed 45ms

Hide
Type "export PS1='\n$ ' && clear"
Enter
Show

Type "cat phishing.eml"
Sleep 400ms
Enter
Sleep 3s

Type "./score.sh phishing.eml"
Sleep 400ms
Enter
Sleep 5s

Hide
Type "clear"
Enter
Show

Type "cat colleague.eml"
Sleep 400ms
Enter
Sleep 3s

Type "./score.sh colleague.eml"
Sleep 400ms
Enter
Sleep 5s
