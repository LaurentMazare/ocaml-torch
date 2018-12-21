This example implements a character-level language model heavily inspired 
by [char-rnn](https://github.com/karpathy/char-rnn).
- For training, the model takes as input a text file and learns to predict the next character following
a given sequence.
- For generation, the model takes as input a seed character sequence, returns the next character distribution
from which a character is randomly sampled and this is iterated to generate a text.

At the end of each training epoch, some sample text is generated and stored in a file in the current directory.

Any text file can be used as an input, as long as it's large enough for training.
A typical example would be the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The training text file should be stored in `data/input.txt`.

Compiling and running the example is done via the following command lines:
```bash
dune build examples/char_rnn/char_rnn.exe
dune exec examples/char_rnn/char_rnn.exe
```
Here is an example of generated data when training on the Shakespeare dataset after a couple epochs. 
```
FFRTEGARY:
I think not to Edward needs to pierce my forth,
I give away before: die, my lady's fee.
Murder me not to bed: and then, makes you wrength,
So worthy speech, misvallest court his true better.

GLOUCESTER:
Thou dost company we hear.

PRINCE EDWARD:
The people must this helper, but us, gall,
I heard of bosom. Ha! quit it, I cleard,
Yet nature's tomb gensly penutal. 'tis begot, by their wills
To under bising him in seems my great queen:
Right crossed me?

BIONDELLO:
What most nothing now?

LEONTES:
Alas, more and Christopy deet-pubuning blame her name.'
My chivallings stand all of heaven; we'll have
Bucklews their friends. What common fire
Edvers thou, or bad modestend themselves with,
Beckinably.

ISABELLA:
In so; or what didst? no, Tybalt?
How came fall? Away, weak, gentle Murderer:
Here wend to do it. B
```
