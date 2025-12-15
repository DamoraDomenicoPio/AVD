To run the program do the following steps:

Go to the folder:
```bash
2024_avd_gr01@PowerEdge-R750xa:~/client_carla$
```

Start Docker - carla server
```bash
docker run --rm --gpus '"device=0"' --net=bridge -p 6000-6001:6000-6001 --name carla_server${USER} -d carla_leaderboard:latest /bin/bash ./CarlaUE4.sh -carla-port=6000 -RenderOffScreen
```

Run:
```bash
./pre_run.sh
./docker_run.sh
```

In another terminal run:

```bash
docker container ls | grep gr01
docker exec -it id bash # where "id" is that of the carla-client
```

Now go to the folder containing server_http.py and run it
```bash
cd team_code
python server_http.py
```

In the first terminal terminal:
```bash
pip install colorama
./team_code/run_test_final.sh
```