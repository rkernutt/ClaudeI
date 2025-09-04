# Claude & I
This repository contains everything you need to build a Docker container running the Claude & I RAG Application
The file upload and chat interfaces are built to leverage Amazon Bedrock and an Elasticsearch Serverless project, If you wish to leverage another LLM service and Elastic locally or in ECH, you will need to modify the environmental variables and prompts.
Out of the box, just modify the .env file with your Elastic and Amazon Bedrock access information, then build your container and you should be good to go.
There are some sample dataset files to upload with info on theme parks, restaurants, hotels, etc to search against.
