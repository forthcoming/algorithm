import logging,random,time,redis
from redis.exceptions import RedisError

class Redlock:
    # https://redis.io/topics/distlock
    # https://github.com/SPSCommerce/redlock-py
    default_retry_count = 3
    default_retry_delay = 0.2
    clock_drift_factor = 0.01
    unlock_script = """
    if redis.call("get",KEYS[1]) == ARGV[1] then
        return redis.call("del",KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, connection_list, retry_count=None, retry_delay=None):
        if not connection_list:
            raise Exception("Failed to connect to the majority of redis servers")
        self.servers = []
        try:
            for connection_info in connection_list:
                if isinstance(connection_info, str):
                    server = redis.StrictRedis.from_url(connection_info)
                elif isinstance(connection_info, dict):
                    server = redis.StrictRedis(**connection_info)
                else:
                    server = connection_info
                self.servers.append(server)
        except Exception as e:
            raise Warning(e)
        self.quorum = len(connection_list) // 2 + 1
        self.retry_count = retry_count or self.default_retry_count
        self.retry_delay = retry_delay or self.default_retry_delay            

    def lock(self, resource, ttl):  # ttl is the number of milliseconds for the validity time.
        assert isinstance(ttl, int), 'ttl {} is not an integer'.format(ttl)
        retry = 0
        val = ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(22))
        # Add 2 milliseconds to the drift to account for Redis expires precision which is 1 millisecond, plus 1 millisecond min drift for small TTLs.
        drift = int(ttl * self.clock_drift_factor) + 2

        while retry < self.retry_count:
            n = 0
            start_time = time.time()
            try:
                for server in self.servers:
                    if server.set(resource, val, nx=True, px=ttl):
                        n += 1
            except RedisError as e:
                logging.exception(e)
            elapsed_time = int((time.time() - start_time) * 1000)
            validity = int(ttl - elapsed_time - drift)
            mutex={'validity':validity, 'resource':resource, 'val':val}
            if validity > 0 and n >= self.quorum:
                return mutex
            else:
                self.unlock(mutex)
                retry += 1
                time.sleep(self.retry_delay)
        return False

    def unlock(self, mutex):
        try:
            for server in self.servers:
                server.eval(self.unlock_script, 1, mutex['resource'], mutex['val'])
        except RedisError as e:
            logging.exception("Error unlocking resource {mutex['resource']} in server {mutex['server']}")

if __name__=='__main__':
    dlm = Redlock([{"host": "localhost", "port": 6379, "db": 0} ])
    my_lock = dlm.lock("my_resource_name",10000)  # 10s
    # do something...
    dlm.unlock(my_lock)
