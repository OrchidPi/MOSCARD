
## SSL cert setup
gsutil cp gs://aif_shared_bucket_p_00/ssl/CertEmulationCA.crt .

## Build Image

REGION="us-central1"
PROJECT_ID=$(gcloud config get project)
PROJECT_HASH="${PROJECT_ID##\*-}"
REPOSITORY="shared-aif-artifact-registry-docker-98be"
IMG=hai_cloudrun_pipeline_rws:v2
TAG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMG"

docker buildx build --platform linux/amd64 --file code/webapp/Dockerfile --tag=$TAG .
## Windows equivalent:
gcloud auth application-default login
gcloud auth login --update-adc
gcloud auth activate-service-account --key-file key.json
gcloud auth login

$REGION = "us-central1"
$PROJECT_ID = gcloud config get-value project
$PROJECT_HASH = $PROJECT_ID -replace '^.*-', ''
$REPOSITORY = "shared-aif-artifact-registry-docker-98be"
$IMG = "moscard_inference:v42"
$TAG = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMG"
echo $REGION $PROJECT_ID $PROJECT_HASH $REPOSITORY $IMG $TAG

docker buildx build --platform linux/amd64 --file webapp/Dockerfile --tag=$TAG .
docker push $TAG

## Steps to containerizing
1) copy Dockerfile from template dir over to a place in this repo that makes sense and adjust the docker command accordingly, read through dockerfile to make sure python is corrcect
2) Set the src folder to the parent of the code to run
3) copy src from the template dir to here

So the pipline with the yaml config calls the webapp\pipeline_input.py module that in turn supplies MOSCARD\bin\test_mimic.py with what it needs to run