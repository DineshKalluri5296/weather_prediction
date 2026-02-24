pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        ECR_REPO = "seattle-ml-app"
        IMAGE_TAG = "${BUILD_NUMBER}"
        ACCOUNT_ID = "079032833883"
        ECR_URI = "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
        FULL_IMAGE_NAME = "${ECR_URI}/${ECR_REPO}:${IMAGE_TAG}"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    credentialsId: 'github-credentials',
                    url: 'https://github.com/DineshKalluri5296/weather_prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m pip install -r requirements.txt'
            }
        }

        stage('Train ML Model') {
            steps {
                sh 'python3 model.py'
            }
        }

        stage('Upload Model to S3') {
           steps {
              withCredentials([[$class: 'AmazonWebServicesCredentialsBinding',
              credentialsId: 'aws-credentials']]) {
              sh '''
              aws s3 cp model.pkl s3://seattle-ml-app/models/${BUILD_NUMBER}/model.pkl
              '''
              }
          }
     }
        
        stage('Build Docker Image') {
            steps {
                sh '''
                echo "Building Docker Image..."
                docker build -t ${FULL_IMAGE_NAME} .
                docker images
                '''
            }
        }

        stage('Login to AWS ECR') {
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding',
                credentialsId: 'aws-credentials']]) {
                    sh '''
                    aws ecr get-login-password --region ${AWS_REGION} | \
                    docker login --username AWS --password-stdin ${ECR_URI}
                    '''
                }
            }
        }

        stage('Push Image to ECR') {
            steps {
                sh '''
                echo "Pushing Image..."
                docker push ${FULL_IMAGE_NAME}
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
                ${FULL_IMAGE_NAME}
                '''
            }
        }
        stage('Deploy Node Exporter') {
            steps {
                sh '''
                docker run -d \
                --name node-exporter \
                -p 9100:9100 \
                prom/node-exporter
                '''
            }
        }

        stage('Deploy Prometheus') {
            steps {
                sh '''
                docker run -d \
                --name prometheus \
                -p 9090:9090 \
                -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
                prom/prometheus
                '''
            }
        }

        stage('Deploy Grafana') {
            steps {
                sh '''
                docker run -d \
                --name grafana \
                -p 3000:3000 \
                grafana/grafana
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
