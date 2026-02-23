pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        ECR_REPO = "seattle-ml-app"
        IMAGE_TAG = "${BUILD_NUMBER}"
        ACCOUNT_ID = "YOUR_AWS_ACCOUNT_ID"
        ECR_URI = "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
    }

    stages {

        stage('Checkout Code') {
           steps {
              git branch: 'main',
              credentialsId: 'github-credentials',
              url: 'https://github.com/DineshKalluri5296/weather_prediction.git'
             }
          }

        stage('Install Python Dependencies') {
            steps {
                sh '''
                pip3 install -r requirements.txt
                '''
            }
        }

        stage('Train ML Model') {
            steps {
                sh '''
                python3 model.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t ${ECR_REPO}:${IMAGE_TAG}
                docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}
                '''
            }
        }

        stage('Login to AWS ECR') {
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding',
                credentialsId: 'aws-credentials']]) {
                    sh '''
                    aws ecr get-login-password --region ${AWS_REGION} | \
                    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
                    '''
                }
            }
        }

        stage('Push Image to ECR') {
            steps {
                sh '''
                docker push ${ECR_URI}:${IMAGE_TAG}
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                sh '''
                docker stop seattle-container || true
                docker rm seattle-container || true
                docker run -d -p 8000:8000 \
                --name seattle-container \
                ${ECR_URI}:${IMAGE_TAG}
                '''
            }
        }

    }

    post {
        success {
            echo "Deployment Successful 🚀"
        }
        failure {
            echo "Pipeline Failed ❌"
        }
    }
}
