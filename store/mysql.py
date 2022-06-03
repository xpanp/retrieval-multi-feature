import pymysql
import struct

TestDB = 'testdb'

class DB():
    def __init__(self, host='49.234.33.206', user='root', password='123456', database='testdb') -> None:
        self.db = pymysql.connect(host=host, user=user, password=password, database=database)
    
    def close(self):
        self.db.close()

    def unpack(self, r):
        # 将bytes数据解包为float list
        f1 = list(struct.unpack('{}f'.format(int(len(r[3])/4)), r[3]))  # vgg16
        f2 = list(struct.unpack('{}f'.format(int(len(r[4])/4)), r[4]))  # lbp
        f3 = list(struct.unpack('{}f'.format(int(len(r[5])/4)), r[5]))  # color
        f4 = list(struct.unpack('{}f'.format(int(len(r[6])/4)), r[6]))  # glcm
        return (r[0], r[1], r[2], f1, f2, f3, f4)

    # return (id filename filepath list list list list)
    def get_one(self, id):
        sql = "SELECT * FROM DATA_VECTOR WHERE ID = %s"
        cursor = self.db.cursor()
        try:
            cursor.execute(sql, (id, ))
            result = cursor.fetchone()
        except pymysql.Error as e:
            print("select fail", e)
        
        if result == None:
            print("no record with id:", id)
            return None
        return self.unpack(result)

    # return filepath
    def get_one_path(self, id):
        sql = "SELECT FILEPATH FROM DATA_VECTOR WHERE ID = %s"
        cursor = self.db.cursor()
        try:
            cursor.execute(sql, (id, ))
            result = cursor.fetchone()
        except pymysql.Error as e:
            print("select fail", e)
        
        if result == None:
            print("no record with id:", id)
            return None
        return result[0]

    # return [(id filename filepath list list list list)]
    def select_all(self):
        sql = "SELECT * FROM DATA_VECTOR ORDER BY ID"
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
        except pymysql.Error as e:
            print("select fail", e)
        
        results_unpack = []
        for r in results:
            unpack_r = self.unpack(r)
            results_unpack.append(unpack_r)
        return results_unpack

    # 传入特征需转化为list类型
    def insert(self, id, filename, filepath, vgg16feat, lbpfeat, colorfeat, glcmfeat):
        # 将float list打包成bytes类型 以便写入blob字段    
        buf1 = struct.pack('%sf' % len(vgg16feat), *vgg16feat)
        buf2 = struct.pack('%sf' % len(lbpfeat), *lbpfeat)
        buf3 = struct.pack('%sf' % len(colorfeat), *colorfeat)
        buf4 = struct.pack('%sf' % len(glcmfeat), *glcmfeat)
        # _binary %s 便于写入blob字段
        sql = "INSERT INTO DATA_VECTOR(FILENAME, FILEPATH, VGG16FEAT, LBPFEAT, COLORFEAT, GLCMFEAT) \
            VALUES (%s, %s, _binary %s, _binary %s, _binary %s, _binary %s)"

        cursor = self.db.cursor()    
        try:
            # 执行sql语句
            cursor.execute(sql, (filename, filepath, buf1, buf2, buf3, buf4))
            # 提交到数据库执行
            self.db.commit()
            print("insert success, ", filename)
        except pymysql.Error as e:
            # 如果发生错误则回滚
            self.db.rollback()
            print("insert failed, ", filename, e)
    