import http.server
import socketserver
import os
import threading
import requests
import random
import time
import redis

# Import Prometheus library
from prometheus_api_client import PrometheusConnect

# Constants
PORT = 31113
WINDOW_SIZE = 10

# Prometheus Configuration
prom = PrometheusConnect(url="http://192.168.56.11:30568")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

class MyHandler(http.server.BaseHTTPRequestHandler):
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    def do_GET(self):
    
        # invoke using curl 127.0.0.1:31113/float-operation
        if self.path == '/float-operation':

            url = "http://192.168.56.11:31112/function/float-operation"
            data = '{"number": 20, "uuid": "1234"}' # example data to send

            # Measure latency
            start_time = time.time()
            response = requests.post(url, data=data)
            end_time = time.time()
            latency = end_time - start_time

            function_name = "float-operation"  # You may extract the function name from the URL if needed

            # Store response time in Redis
            self.update_response_times(function_name, latency)

            if response.status_code == 200:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                message = f"Fibonacci value: {response.text}\n"
                print(latency)
            else:
                self.send_response(500)
                message = "Error: Internal Server Error"
            self.wfile.write(bytes(message, "utf8"))
        else:
            self.send_error(404)

    def update_response_times(self, function_name, latency):
        # Append the new latency to the list in Redis
        self.redis_client.rpush(function_name, latency)

        # Trim the list to maintain a window size of 10
        self.redis_client.ltrim(function_name, -WINDOW_SIZE, -1)

        # Get the response times from Redis
        response_times = self.redis_client.lrange(function_name, 0, -1)

        # Calculate and print average response time
        if len(response_times) < WINDOW_SIZE:
            # Adjust window size if the number of response times is less than the specified window size
            window_size = len(response_times)
        else:
            window_size = WINDOW_SIZE
        
        if window_size > 0:
            avg_response_time = sum(map(float, response_times)) / window_size
            print(f"Average response time for {function_name}: {avg_response_time} seconds")
        else:
            print(f"No response times available for {function_name}")

def main():
    # Start the HTTP server
    httpd = ThreadedHTTPServer(("", PORT), MyHandler)
    print("serving at port", PORT)
    httpd_thread = threading.Thread(target=httpd.serve_forever)
    httpd_thread.daemon = True
    httpd_thread.start()

    # Start a daemon or thread that stores the active time of VMs in the Kubernetes cluster
    active_vm_thread = threading.Thread(target=update_active_vms)
    active_vm_thread.daemon = True
    active_vm_thread.start()
    while True:
        pass


def update_active_vms():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    ready_nodes = []
    # Get ready nodes in the cluster
    cpu_query = 'sum(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (node) * 1000'

    available_cpu = prom.custom_query(cpu_query)

    for dic1 in available_cpu:
        if dic1['metric']['node'] == 'k3smaster':
            continue
        else:
            ready_nodes.append(dic1['metric']['node'])
    
    # Initialize all nodes in the database with a default value of 0
    for node in ready_nodes:
        redis_key = f"active_time:{node}"
        redis_client.set(redis_key, 0)
    
    while True:
        # Get active status of each node
        active_status_results = prom.custom_query('count(kube_pod_info{namespace="openfaas-fn"}) by (node)')
        for node in ready_nodes:
            node_dict = find_node(active_status_results, node)
            if node_dict is not None:
                # The node is active so add 2 seconds to active time of the node
                update_node_active_time(node, 2)
            else:
                # The node is not active, do nothing
                pass

        # Wait for 2 seconds before checking again
        time.sleep(2)

def find_node(data, node_name):
    for item in data:
        if 'node' in item['metric'] and item['metric']['node'] == node_name:
            return item
        elif 'instance' in item['metric'] and item['metric']['instance'] == node_name:
            return item
    return None

def update_node_active_time(node_name, increment):
    # Connect to Redis
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    # Increment the active time of the node in the Redis database
    redis_key = f"active_time:{node_name}"
    print(redis_key)
    redis_client.incrby(redis_key, increment)

if __name__ == "__main__":
    main()


redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

