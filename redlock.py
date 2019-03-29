import logging,time,redis,random,threading
from redis.exceptions import RedisError
from os import urandom
from hashlib import sha1

class Redlock:
    # KEYS[1] - lock name
    # ARGS[1] - token
    # return 1 if the lock was released, otherwise 0
    unlock_script = """
        if redis.call("get",KEYS[1]) == ARGV[1] then
            return redis.call("del",KEYS[1])
        else
            return 0
        end
    """

    def __init__(self, connection_list, name, ttl=10, blocking_timeout=20, thread_local=True):
        '''
        reference:
            https://redis.io/topics/distlock
            https://github.com/SPSCommerce/redlock-py
            https://github.com/andymccurdy/redis-py/blob/master/redis/lock.py

        ttl is the number of milliseconds for the validity time.如果担心加锁业务在锁过期时还未执行完,可以给该业务加一个定时任务,定时检测业务是否还存活并决定是否给锁续期
        thread_local indicates whether the lock token is placed in thread-local storage. By default, the token is placed in thread local storage so that a thread only sees its token, 
        not a token set by another thread. Consider the following timeline:
        time: 0, thread-1 acquires `my-lock`, with a timeout of 5 seconds.thread-1 sets the token to "abc"
        time: 1, thread-2 blocks trying to acquire `my-lock` using the Lock instance.
        time: 5, thread-1 has not yet completed. redis expires the lock key.
        time: 5, thread-2 acquired `my-lock` now that it's available.thread-2 sets the token to "xyz"
        time: 6, thread-1 finishes its work and calls release(). if the token is *not* stored in thread local storage,then thread-1 would see the token value as "xyz" and would be able to successfully release the thread-2's lock.
        In some use cases it's necessary to disable thread local storage. 
        For example, if you have code where one thread acquires a lock and passes that lock instance to a worker thread to release later. 
        If thread local storage isn't disabled in this case, the worker thread won't see the token set by the thread that acquired the lock. 
        Our assumption is that these cases aren't common and as such default to using thread local storage.
        '''
        assert isinstance(ttl, int), 'ttl {} is not an integer'.format(ttl)
        self.servers = []
        for connection_info in connection_list:
            if isinstance(connection_info, str):
                server = redis.Redis.from_url(connection_info)
            elif isinstance(connection_info, dict):
                server = redis.Redis(**connection_info)
            else:
                server = connection_info
            server.script_load(self.unlock_script)
            self.servers.append(server)
        self.quorum = len(connection_list) // 2 + 1
        self.name = name
        self.ttl = ttl
        self.blocking_timeout = blocking_timeout
        self.local = threading.local() if thread_local else type('dummy',(),{})
        self.sha = sha1(bytes(self.unlock_script,encoding='utf8')).hexdigest()  # script_load其实已经返回了lua脚本对应的sha1值
    
    def __enter__(self):
        if self.lock():
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.unlock():
            print('exc_type: {}, exc_value: {}, traceback: {}.'.format(exc_type, exc_value, traceback))
        return True
        
    def lock(self):
        self.local.token = urandom(16)
        drift = int(self.ttl * .01) + 2  # Add 2 milliseconds to the drift to account for Redis expires precision which is 1 millisecond, plus 1 millisecond min drift for small TTLs.
        start_time = time.time()
        stop_at = start_time + self.blocking_timeout
        while start_time < stop_at:
            n = 0
            try:
                for server in self.servers:
                    if server.set(self.name, self.local.token, nx=True, px=self.ttl):
                        n += 1
            except RedisError as e:
                logging.exception(e)
            elapsed_time = int((time.time() - start_time) * 1000)
            validity = int(self.ttl - elapsed_time - drift)
            if validity > 0 and n >= self.quorum:
                return True
            else:  # 如果锁获取失败应立马释放获取的锁定
                self.unlock()
                time.sleep(random.uniform(0,.4))  # a random delay in order to try to desynchronize multiple clients trying to acquire the lock for the same resource at the same time
            start_time = time.time()
        raise Exception("lock timeout")

    def unlock(self):
        try:
            for server in self.servers:
                server.evalsha(self.sha, 1, self.name, self.local.token)  # 原子操作
            return True
        except RedisError as e:
            logging.exception("Error: unlocking lock {}".format(self.name))

    def do_something(self):
        print('Im doing something')

if __name__=='__main__':
    connection_list = [
        {"host": "localhost", "port": 6379, "db": 0},
    ]
    with Redlock(connection_list,'my_resource_name',10) as dlm:
        dlm.do_something()
