# Hallucination Resistance in Large Language Models


## Getting started

1. Generate ssh key using`ssh-keygen`

2. Setup ssh key for your account
https://ml-contest.gitlab.yandexcloud.net/-/user_settings/ssh_keys

Use .pub file that you generated on previous step

3. `ssh-add <private key>`
private key is typically `~/.ssh/id_rsa`


4. `git clone git@ml-contest.gitlab.yandexcloud.net:problems/<your repo path>.git`

5. Write your soultion.
The example solution could be seen in files solution.py. The environment that it will use is provided in Dockerfile. You can customize Dockerfile to your needs. If you need other packages, just add them there.
We will run `docker build` inside your repository, so all the files will be available by the same relative paths.

6. After you completed your solution, you can submit it using one of the two options.

## Submitting your solution using Git LFS (slow)
```bash
git lfs install
git lfs track ./big_model_weights/**
git add .
git commit -m "Solution"
git push
```
Then click use the button to submit your task in contest.yandex.ru
Remember, that there is a limit of to submissions per day

## Submitting your solution using DVC (fast, experimental)
This method uses dvc instead of LFS https://dvc.org/

1. Run ./init_dvc.sh. This will allow you to properly initiate dvc (data version control) that we will use to store large files (like model weigths)
2. To submit your solution use following commands:
```bash
dvc add <model checkpoint path> # important: checkpoint should be in the same directory as other project files
dvc push  # upload model weights to S3
git add . # commit the rest of files
git commit -m "Solution"
git push
```
Then click use the button to submit your task in contest.yandex.ru
Remember, that there is a limit of to submissions per day

