# build image
docker build -t durianclassifierapi:v3 .

# try running the image
docker run -p 5000:5000 durianclassifierapi:v3 

# tag the image version (change v1 to v2, etc)
docker tag durianclassifierapi:v3 thomasibudiman/durian-classifier:v3

# login to docker from cmd
docker login

# push to docker hub
docker push thomasibudiman/durian-classifier:v3

# push to GitHub
git add .
git commit -m "update"
git push origin main


# To download ("--name" specify the name locally, can be anything but it will get from thomasibudiman/durian-classifier:v3)
docker stop durianclassifierapi || true
docker rm durianclassifierapi || true
docker pull thomasibudiman/durian-classifier:v3
docker run -d -p 80:5000 \
            --name durianclassifierapi \
            --cpus="0.5" \
            thomasibudiman/durian-classifier:v3