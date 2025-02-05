trigger:
  branches:
    include:
      - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.x'
      addToPath: true

  - script: |
      python -m pip install --upgrade pip
      pip install -r requirements.txt
    displayName: 'Install dependencies'

  - script: |
      python train_gan.py
    displayName: 'Run GAN Training'
    
  - publish: $(Build.ArtifactStagingDirectory)
    artifact: drop






task: AzureBlobStorage@1
    inputs:
      connectionString: $(connectionString)  # Connection string to your Azure Blob Storage
      containerName: 'generated-images'
      filePath: '$(Build.ArtifactStagingDirectory)/generated_images/*.png'
      blobPrefix: 'images/'

















