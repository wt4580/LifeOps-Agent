pipeline {
    agent { label 'docker' }
    options { timestamps(); skipDefaultCheckout(true); disableConcurrentBuilds(); timeout(time: 30, unit: 'MINUTES') }
    parameters {
        string(name: 'PIP_INDEX_URL', defaultValue: 'https://pypi.org/simple', description: 'HTTPS Python index; China: https://pypi.tuna.tsinghua.edu.cn/simple')
        booleanParam(name: 'DEPLOY_KIND', defaultValue: false, description: 'Deploy to local kind dev / namespace lifeops-demo')
    }
    environment {
        TEST_IMAGE = "lifeops-agent:test-${BUILD_NUMBER}"
        APP_IMAGE = "lifeops-agent:build-${BUILD_NUMBER}"
        TEST_CONTAINER = "lifeops-test-${BUILD_NUMBER}"
        SMOKE_CONTAINER = "lifeops-smoke-${BUILD_NUMBER}"
    }
    stages {
        stage('Checkout') { steps { checkout scm } }
        stage('Build test image') {
            steps { sh 'docker build --target test --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" -t "$TEST_IMAGE" .' }
        }
        stage('Unit and contract evaluation') {
            steps {
                sh '''
                    docker create --name "$TEST_CONTAINER" "$TEST_IMAGE" sh -ec '
                        python -m ruff check --select E9,F63,F7,F82 app tests scripts
                        python -m compileall -q app scripts
                        python -m pytest tests -q --junitxml=/reports/unit.xml
                        python -m app.eval.run_e2e --output /reports/e2e.json
                    '
                    docker start -a "$TEST_CONTAINER"
                    test "$(docker inspect -f '{{.State.ExitCode}}' "$TEST_CONTAINER")" = 0
                '''
            }
        }
        stage('Build and smoke') {
            steps {
                sh '''
                    docker build --target production --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" -t "$APP_IMAGE" .
                    docker run -d --name "$SMOKE_CONTAINER" --read-only --tmpfs /tmp:rw,size=64m \
                        -e DEMO_MODE=true -e DEMO_SEED=true -e BACKGROUND_LEARNING_ENABLED=false \
                        -e SCHEDULER_ENABLED=false "$APP_IMAGE"
                    docker exec "$SMOKE_CONTAINER" python scripts/smoke.py --url http://127.0.0.1:5000
                '''
            }
        }
        stage('Deploy kind') {
            when { expression { params.DEPLOY_KIND } }
            steps {
                sh '''
                    kind load docker-image "$APP_IMAGE" --name dev
                    kubectl --context kind-dev apply -f deploy/kind/demo.yaml
                    kubectl --context kind-dev -n lifeops-demo set image deployment/lifeops-demo lifeops="$APP_IMAGE"
                    kubectl --context kind-dev -n lifeops-demo rollout status deployment/lifeops-demo --timeout=120s
                    kubectl --context kind-dev -n lifeops-demo exec deployment/lifeops-demo -- python scripts/smoke.py --url http://127.0.0.1:5000
                '''
            }
        }
    }
    post {
        always {
            sh '''
                mkdir -p artifacts
                if docker inspect "$TEST_CONTAINER" >/dev/null 2>&1; then
                    docker cp "$TEST_CONTAINER:/reports/." artifacts/
                    docker rm -f "$TEST_CONTAINER"
                fi
                if docker inspect "$SMOKE_CONTAINER" >/dev/null 2>&1; then
                    docker logs "$SMOKE_CONTAINER" > artifacts/smoke.log 2>&1
                    docker rm -f "$SMOKE_CONTAINER"
                fi
            '''
            junit allowEmptyResults: true, testResults: 'artifacts/unit.xml'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'artifacts/*'
        }
    }
}
