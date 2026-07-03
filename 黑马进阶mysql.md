# 多表查询

## 分类

内连接：两表连接之后的部分表信息。

左外连接：包含左表的所有信息和右表的部分信息。

右外连接：右表的全部字段和左表的部分字段。

自连接：左右两张表都是自己。如员工表，里面上下级关系，可以设置一个parent_id字段记录，如张三，ID为1，李四，ID为2，parent_id为1，此时要查询李四的所有信息及其上级的名字，就需要先查到李四的所有信息，再通过parent_id再查一次表。

```
SELECT 
    e.*, 
    m.name AS manager_name
FROM 
    emp e
LEFT JOIN 
    emp m ON e.parent_id = m.id;
```

子查询：先按某种条件查询得到一个临时表，再在外面套一层查询。

```
SELECT 
    e.*,
    (SELECT name FROM emp WHERE id = e.parent_id) AS manager_name
FROM 
    emp e;
```

## 子查询掌握要点

一个查询有几种结果：

- 标量子查询：查询出来的为一个值，此时这个子查询就相当于一个值，可以作为另一个查询的条件使用。

  ```sql
  select name from emp where id=3;
  
  select id from emp where name=(select name from emp where id=3);
  ```

- 列子查询：查询出来的为某个字段的多行数据，此时可以用作范围查询的条件

  ```sql
  select id from dept where name = '销售部' or name = '市场部';
  
  select * from emp where dept_id in (select id from dept where name = '销售部' or
  name = '市场部');
  ```



# 事务

## 控制事务指令

```sql
-- 查看和设置事务是否自动提交，默认为1，即为自动提交
SELECT @@autocommit ;
SET @@autocommit = 0 ;

-- 开启事务
start transaction 或 BEGIN ;
-- 1. 查询张三余额
select * from account where name = '张三';
-- 2. 张三的余额减少1000
update account set money = money - 1000 where name = '张三';
-- 3. 李四的余额增加1000
update account set money = money + 1000 where name = '李四';
-- 如果正常执行完毕, 则提交事务
commit;
-- 如果执行过程中报错, 则回滚事务
-- rollback;
```

## ACID

- 原子性（Atomicity）：事务是不可分割的最小操作单元，要么全部成功，要么全部失败。

- 一致性（Consistency）：事务完成时，必须使所有的数据都保持一致状态。
- 隔离性（Isolation）：数据库系统提供的隔离机制，保证事务在不受外部并发操作影响的独立环境下运行。

- 持久性（Durability）：事务一旦提交或回滚，它对数据库中的数据的改变就是永久的。

## 并发事务问题

- 脏读：两个线程ab同时开启事务，此时b对ID为2的员工信息进行了修改，a查询到了修改的数据，但是b出错回滚了，也就导致a查询到了脏数据。
- 不可重复读：两个线程ab同时开启事务，a线程先查询了ID为2的员工信息，b对该员工信息进行了修改，此时a再次查询，和之前查出的数据不一致了。也就是一个事务中两次查询同一个数据不一致，就称为不可重复读。
- 幻读：两个线程ab同时开启事务，a在插入ID为3的员工信息前，先判断是否有ID为3的用户，执行语句之后发现数据库没有，在判断出没有对应信息和真正插入信息之间，b先一步对ID为3的信息进行了插入，a由于不会再次判断，而是直接插入，就会遇到查询时明明没有，但是插入时存在的幻觉情况。

## 事务隔离级别

不同种的隔离级别下会有不同的问题，越低级的隔离级别效率越高，但是出现并发事务问题的风险越大。

![image-20250601111633527](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20250601111633527.png)

- 读未提交：可以读取其他事务未提交的数据，可能出现脏读、不可重复读和幻读三种问题。
- 读已提交：只能读取其他事务已经提交的数据。
- 可重复读：同一个事务内，同一个数据一定是一致的，两次查询结果不会不一样。
- 串行化：

具体实现原理应该在锁的部分。

### 指令

1.  查看事务隔离级别

   ```sql
   SELECT @@TRANSACTION_ISOLATION;
   ```

2. 设置事务隔离级别

   ```sql
   SET [ SESSION | GLOBAL ] TRANSACTION ISOLATION LEVEL { READ UNCOMMITTED |
   READ COMMITTED | REPEATABLE READ | SERIALIZABLE }
   ```

   - session和global的作用范围指的是会话，也可以说是连接，如果用 `mysql -u root -p` 打开两个终端，它们是两个独立的会话（连接）。而navicat相当于图形化界面，他也是一次连接，在navicat上使用session设置也就相当于设置了navicat上这个连接的事务隔离级别。
   - 对于Java项目对mysql的连接，要看使用的是哪种连接方式，若是使用手动在查询之后释放连接的方式，那么每次查询都会创建一个新的连接。而若是使用连接池进行连接，则根据连接池本身和如何使用连接池来判断。微服务多模块的连接情况也要看连接池使用情况。
   - navicat在断开连接重新连接后，也会被视为一个新的连接。



# 存储引擎

- 存储引擎就是存储数据、建立索引、更新/查询数据等技术的实现方式 。存储引擎是基于表的，而不是基于库的，所以存储引擎也可被称为表类型。我们可以在创建表的时候，来指定选择的存储引擎，如果没有指定将自动选择默认的存储引擎。mysql5.5之后默认存储引擎为innodb
- **可以通过指令查询当前数据库支持的存储引擎和某个表的建表语句**（包含该表对应的存储引擎）。

## 常用存储引擎对比

1. innodb：支持事务、行级锁和外键约束。适用于读多写少，安全性要求高的情况。==为mysql5.5之后默认存储引擎==
2. mylsam：不支持事务、支持表锁，不支持外键约束。适用于写多同时允许一定的错误的情况。
3. memory：使用内存存储，受到断电等因素的约束，常用于临时表存储，但现在基本被redis替代。

### 文件存储

- sdi存储的为表结构，如各个字段的属性、类型等信息。
- 还需要存储数据和索引相关信息。
- 不同存储引擎存储结构不同，如innodb文件只有后缀为idb的，存储所有内容，而mylsam则分为sdi、myd和myi三种后缀文件，分别存储表结构、数据和索引相关信息。

### 对比

- InnoDB: 是Mysql的默认存储引擎，支持事务、外键。如果应用对事务的完整性有比较高的要求，在并发条件下要求数据的一致性，数据操作除了插入和查询之外，还包含很多的更新、删除操作，那么InnoDB存储引擎是比较合适的选择。
- MyISAM ： 如果应用是以读操作和插入操作为主，只有很少的更新和删除操作，并且对事务的完整性、并发性要求不是很高，那么选择这个存储引擎是非常合适的。
- MEMORY：将所有数据保存在内存中，访问速度快，通常用于临时表及缓存。MEMORY的缺陷就是对表的大小有限制，太大的表无法缓存在内存中，而且无法保障数据的安全性。





# 索引

- 索引是帮助数据库高效获取数据的数据结构。

- 索引是在存储引擎层实现的，不同存储引擎支持不同的索引。

  ![image-20250601194255853](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20250601194255853.png)

- ==索引使用和失效的情况主要关注where条件；而是否为覆盖索引（是否需要回表查询）则主要注意需要返回的列是否能在使用的索引上直接全部找到。==

- 联合索引的最左原则看的是where后是否使用，而不看条件使用顺序，比如name_phone联合索引，若是where条件为 where phone = '123' and name = '李某'；phone先而name后，实际上也使用了联合索引，并且为覆盖索引。

- 而对于使用order by时，则必须和索引顺序对应，并且order有升序和降序两种，在创建时也可以指定，若是需要的顺序和索引创建时指定的不同，会导致索引失效，从而using filesort；

  ```sql
  create index idx_user_age_phone_ad on tb_user(age asc ,phone desc);
  ```

  ```sql
  //两个字段排序时，先按照第一个字段排序，第一个字段相同则按照第二个字段排序
  explain select id,age,phone from tb_user order by age asc , phone desc ;
  ```

  

