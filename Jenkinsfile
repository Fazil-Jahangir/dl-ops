pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t nlp-fastapi-app .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh '''
                docker stop nlp-container || true
                docker rm nlp-container || true
                docker run -d -p 8000:8000 --name nlp-container nlp-fastapi-app
                '''
            }
        }
    }
}
