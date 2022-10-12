###
# this shell script connects to container's bash
###

# get id of running container with image name "stable-baselines_server_1_1"
container_id=$(docker ps | grep stable-baselines_server_1_1 | awk '{print $1}')
echo "container id: $container_id"
# connect to container
docker exec -it $container_id /bin/bash
