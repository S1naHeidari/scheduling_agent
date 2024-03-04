import time
import random
import json
from kubernetes import client, config, watch
from kubernetes.client.models.v1_container_image import V1ContainerImage
from prometheus_api_client import PrometheusConnect
import Scheduler
import numpy as np



prom = PrometheusConnect(url="http://192.168.56.11:30568")

cpu_query = 'sum(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (node) * 1000'
memory_query = 'sum(node_memory_MemAvailable_bytes) by (node) / 1024 / 1024'

def names(self, names):
	self._names = names
V1ContainerImage.names = V1ContainerImage.names.setter(names)

config.load_kube_config('config')
v1 = client.CoreV1Api()


scheduler_name = "rl-scheduler"

def nodes_available(cpu_req, mem_req):
	ready_nodes = []
	
	available_cpu = prom.custom_query(cpu_query)
	available_memory = prom.custom_query(memory_query)
	for dic1, dic2 in zip(available_cpu, available_memory):
		if dic1['metric']['node'] == 'k3smaster':
			continue
		if float(dic1['value'][1]) > cpu_req and float(dic2['value'][1]) > mem_req:
			ready_nodes.append(dic1['metric']['node'])
	
	return ready_nodes

def scheduler(name, node, namespace = 'openfaas-fn'):
	print(name, node)
	#try:
	#except ValueError:
	#	pass
	# target_sent = client.V1ObjectReference()
	# #target_sent.namespace = 'openfaas-fn'
	# target_sent.kind = "Node"
	# target_sent.apiVersion = "v1"
	# target_sent.api_version = "v1"

	# target_sent.name = node
	# meta = client.V1ObjectMeta()
	# meta.name = name
	# body = client.V1Binding(target = target_sent)

	body = {
    'api_version': 'v1',
    'kind': 'Binding',
    'metadata': {
        'name': name,
        'namespace': 'openfaas-fn'
    },
    'target': {
        'api_version': 'v1',
        'kind': 'Node',
        'name': node
    }}

	# body.target = target_sent
	# body.metadata = meta
	# body.kind = 'Binding'
	# body.target.kind = 'Node'
	# try:
	result = v1.create_namespaced_binding(namespace, body, _preload_content=False)
	response_dict = json.loads(result.data.decode('utf-8'))


	if response_dict['status'] == 'Success':
    		return True
	else:
    		return False
    		
	# return result
	# except:
	# 	pass
	# return False

def main():
    
	seed = 777
	np.random.seed(seed)
	Scheduler.seed_torch(seed)

	# # parameters
	num_frames = 10000
	memory_size = 2000
	batch_size = 4
	target_update = 100
	epsilon_decay = 1 / 2000
	β = 0.5

	env = Scheduler.KubeEnvironment(beta=β)
	#env = Scheduler.CarRentalEnvironment()


	agent = Scheduler.DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)

	w = watch.Watch()
	for event in w.stream(v1.list_namespaced_pod, "openfaas-fn"):
		if event['object'].status.phase == "Pending" and event['object'].spec.scheduler_name == scheduler_name:
			requests = event['object'].spec.containers[0].resources.requests
			cpu_request = int(requests['cpu'].replace('m',''))
			memory_request = int(requests['memory'].replace('Mi',''))
			# res = scheduler(event['object'].metadata.name,"k3sg2node1")
			try:
    				res = agent.schedule(event['object'].spec.containers[0].name, event['object'].metadata.name, cpu_request, memory_request, False)
			except client.rest.ApiException as e:
				print(json.loads(e.body)['message'])

if __name__ == '__main__':
	main()
