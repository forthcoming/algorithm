import logging,time,redis,random
from redis.exceptions import RedisError
from os import urandom

class Redlock:
    # https://redis.io/topics/distlock
    # https://github.com/SPSCommerce/redlock-py
    unlock_script = """
    if redis.call("get",KEYS[1]) == ARGV[1] then
        return redis.call("del",KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, connection_list, retry_count=None):
        self.servers = []
        for connection_info in connection_list:
            if isinstance(connection_info, str):
                server = redis.Redis.from_url(connection_info)
            elif isinstance(connection_info, dict):
                server = redis.Redis(**connection_info)
            else:
                server = connection_info
            self.servers.append(server)
        self.quorum = len(connection_list) // 2 + 1
        self.retry_count = retry_count or 3

    def lock(self, resource, ttl):  # ttl is the number of milliseconds for the validity time.
        assert isinstance(ttl, int), 'ttl {} is not an integer'.format(ttl)
        retry = 0
        val = urandom(16)
        # Add 2 milliseconds to the drift to account for Redis expires precision which is 1 millisecond, plus 1 millisecond min drift for small TTLs.
        drift = int(ttl * .01) + 2

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
                time.sleep(random.uniform(0,.4))  # a random delay in order to try to desynchronize multiple clients trying to acquire the lock for the same resource at the same time
        raise Exception("lock failed")

    def unlock(self, mutex):
        try:
            for server in self.servers:
                server.eval(self.unlock_script, 1, mutex['resource'], mutex['val'])  # 原子性操作
        except RedisError as e:
            logging.exception("Error unlocking resource {mutex['resource']} in server {mutex['server']}")

if __name__=='__main__':
    dlm = Redlock([{"host": "localhost", "port": 6379, "db": 0} ])
    my_lock = dlm.lock("my_resource_name",10000)  # 10s
    # do something...
    dlm.unlock(my_lock)
