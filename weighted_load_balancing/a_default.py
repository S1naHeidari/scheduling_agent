import http.server
import socketserver
import os
import threading
import requests
import random
import time
import redis

PORT = 31113
WINDOW_SIZE = 10

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

class MyHandler(http.server.BaseHTTPRequestHandler):
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    def do_GET(self):
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

httpd = ThreadedHTTPServer(("", PORT), MyHandler)
print("serving at port", PORT)
httpd.serve_forever()
