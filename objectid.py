import binascii,calendar,datetime,hashlib,os,random,socket,struct,threading,time

'''
代码改编自pymongo的bson.objectid. from bson import objectid
mongodb的每一条插入语句都会包含if "_id" not in document:document["_id"] = ObjectId()语句
'''
def _machine_bytes():
    machine_hash = hashlib.md5()
    machine_hash.update(socket.gethostname().encode())
    return machine_hash.digest()[0:3]

class ObjectId:
    # 此处的代码会在类加载时执行一次,类实例化时不再执行
    _inc = random.randint(0, 0xFFFFFF)
    _inc_lock = threading.Lock()
    _machine_bytes = _machine_bytes()
    __slots__ = ('__id')

    def __init__(self, oid=None):
        """
        Note that the timestamp and counter fields must be stored big endian,cause they are compared byte-by-byte and we want to ensure a mostly increasing order
        An ObjectId is a 12-byte unique identifier consisting of:
        4-byte value representing the seconds since the Unix epoch,
        3-byte machine identifier,
        2-byte process id, 
        3-byte counter, starting with a random value.
        By default,ObjectId() creates a new unique identifier. The optional parameter oid can be any 12 class:bytes.
        For example, the 12 bytes b'foo-bar-quux' do not follow the ObjectId specification but they are acceptable input:
        ObjectId(b'foo-bar-quux') # 666f6f2d6261722d71757578
        oid can also be class:str of 24 hex digits:
        ObjectId('0123456789ab0123456789ab') # 0123456789ab0123456789ab
        """
        if oid is None: # Generate a new value for this ObjectId.
            oid = struct.pack(">i", int(time.time()))  # >: big-endian   i: int 
            oid += ObjectId._machine_bytes
            oid += struct.pack(">H", os.getpid() & 0xFFFF)  # H: unsigned short
            with ObjectId._inc_lock:              # 确保相同进程的同一秒产生的ID也是不同的,前提是相同进程同一秒产生的ID不能超过2^24
                oid += struct.pack(">i", ObjectId._inc)[1:4]  # _inc只有3byte长度so高位会被填充成0
                ObjectId._inc = (ObjectId._inc + 1) & 0xFFFFFF
            self.__id = oid
        elif isinstance(oid, bytes) and len(oid) == 12:
            self.__id = oid
        elif isinstance(oid, str):
            if len(oid) == 24:
                try:
                    self.__id = bytes.fromhex(oid)  # 16进制字符串转换成转换成16进制数,再转换成b串
                except (TypeError, ValueError):
                    print('oid is not a valid ObjectId, it must be a 12-byte input or a 24-character hex string')
            else:
                print('oid is not a valid ObjectId, it must be a 12-byte input or a 24-character hex string')
        else:
            raise TypeError(f"id must be an instance of (bytes,str),not {type(oid)}")

    @classmethod
    def from_datetime(cls, generation_time):
        """
        warning:
           It is not safe to insert a document containing an ObjectId generated using this method. 
           This method deliberately eliminates the uniqueness guarantee that ObjectIds generally provide. 
           ObjectIds generated with this method should be used exclusively in queries.
        generation_time will be converted to UTC. Naive datetime instances will be treated as though they already contain UTC.
        This method is useful for doing range queries on a field containing class:ObjectId instances.
        An example using this helper to get documents where _id was generated before January 1, 2010 would be:
        >>> gen_time = datetime.datetime(2010, 1, 1)
        >>> dummy_id = ObjectId.from_datetime(gen_time)
        >>> result = collection.find({"_id": {"$lt": dummy_id}})
        generation_time: 
            class:datetime.datetime to be used as the generation time for the resulting ObjectId.
        """
        timestamp = calendar.timegm(generation_time.timetuple())
        oid = struct.pack(">i", int(timestamp)) + b"\x00\x00\x00\x00\x00\x00\x00\x00"
        return cls(oid)

    @property
    def generation_time(self):
        timestamp = struct.unpack(">i", self.__id[0:4])[0]
        return datetime.datetime.fromtimestamp(timestamp)

    def __str__(self):
        return binascii.hexlify(self.__id).decode() # 将byte类型转换成byte十六进制类型,再转换成str类型,eg: binascii.hexlify(b'aA').decode()输出6141

    def __hash__(self):  # Get a hash value for this class:ObjectId.
        return hash(self.__id)

    def __lt__(self, other):
        if isinstance(other, ObjectId):
            return self.__id < other.__id
        return NotImplemented
