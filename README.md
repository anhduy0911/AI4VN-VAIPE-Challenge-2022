## Preparation

The file `competition.yaml` contains all the metadata about the competition, including the phases, start dates, end dates etc.., modify the file accordingly.

Note that the content of the *.html can also be editted with a WYSIWYG editor after uploading to the AIHUB platform

## Scoring Program

To submit the results, the user is required to submit a zip file that contains a `results.csv`. 

Each phase will have a ground truth file e.g `warm_up/ground_truth.csv`, `public/ground_truth.csv`, `private/ground_truth.csv`, 

AIHUB will execute `evaluate.py` and pass in the paths to the ground_truth file and the user submitted file. Currently, the example `evaluate.py` calculates ROC AUC metric from the ground truth and the submitted file, modify accordingly if other metrics are required.

## Packaging
To create .zip package, run the following command

```
./package.sh
```

After packaging, upload the file competition.zip to the AIHUB platform
