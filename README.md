# Deep Explanation Penalization (CDEP)
This repository is a reimplementation of [deep-explanation-penalization](https://github.com/laura-rieger/deep-explanation-penalization) in
`TensorFlow 2.4`.

## Using DVC

### Initializing DVC
```bash
dvc remote add origin --local https://dagshub.com/<DAGsHub-user-name>/hello-world.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DAGsHub-user-name>
dvc remote modify origin --local ask_password true
```

### Adding Data
```bash
dvc add data
```
The above command creates `data.dvc`, which should be committed with `git add data.dvc`.