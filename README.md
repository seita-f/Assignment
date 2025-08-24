# Environment
- MacOS Montery (Version 12.7.6)
- VSCode
- github for version control (private repo)

# Setup
1. Build an image, create a container
Run under the assignment directory
```
$ cd Assignment
$ docker compose up -d  
```

2. Run a container
```
$ docker run -it --rm assignment:v1
```

note: mount fof dev
```
docker run -it --rm -v /Users/fujiwaraseita/Desktop/Assignment:/Assignment assignment:v1
```

```
python -m src.data.load_dataset
python -m src.features.main
```

# Training


# Test (Inference)




