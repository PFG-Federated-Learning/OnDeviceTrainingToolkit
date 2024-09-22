from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

def server_fn(context: Context) -> ServerAppComponents:
    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(config=config)

def create_server_app():
    return ServerApp(server_fn=server_fn)
