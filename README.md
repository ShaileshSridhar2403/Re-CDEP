# Deep Explanation Penalization (CDEP)
This repository is a reimplementation of [deep-explanation-penalization](https://github.com/laura-rieger/deep-explanation-penalization) in
`TensorFlow 2.4`.

## Using DVC

### Initializing DVC
The following commands should be run only for the first time:
```bash
dvc remote add origin https://dagshub.com/midsterx/deep-explanation-penalization-keras.dvc
dvc remote add origin --local https://dagshub.com/midsterx/deep-explanation-penalization-keras.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DAGsHub-user-name>
dvc remote modify origin --local ask_password true
dvc remote modify origin --local password <your_token>

dvc push -r origin
```

### Adding Data
```bash
dvc add data
```
The above command creates `data.dvc`, which should be committed with:
```bash
git add data.dvc
```