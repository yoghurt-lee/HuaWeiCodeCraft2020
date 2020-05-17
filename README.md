# HuaWeiCodeCraft2020
2020华为软挑初赛武长赛区第一，复赛武长赛区A榜第二解决方案 

队伍名: 左家垅反洗钱小分队,来自中南大学，队长是Hewitt，我是队员木子烧饼，另一位队员是TXL，非常感谢队长两个月来的辛苦付出！

实际上最后一版代码线下找环优化了约 0.5 秒(1963W数据集)， 输出优化了约 0.06 秒，可惜一直没有成功提交(2020/5/15最后一次提交文件名写错)，不得不说是一种遗憾。

* 运行设置
  1. 系统环境: **Linux** 。
  2. 更改函数 **readAndDraw()** 中的输入文件路径。
  3. 编译命令: ``` g++ -O3 文件名.cpp -o test -lpthread -fpic ```
  4. 运行命令: ``` time ./test ```
  
* 输入方面:
  1. 读取文件时按照字符出现频率进行判断，使用 **while** 结构代替 **if continue** 结构.
  2. 设置阈值top ，当前ID值≤top时使用Bool数组进行去重， >top时直接加入结点数组，之后对节点数组整个进行排序。
  3. 并行排序，使用类似快排的思想对节点数组进行划分，划分到设定阈值大小时放入任务队列等待子线程进行处理。
  4. 映射数字时根据阈值top , ≤top直接使用数组进行映射，>top时使用二分法进行查找。


* 找环方面:
  1. 我们一直使用的 **3+6 + 2+5** 的算法，最后两天优化了**3+4**算法， 最终版本采用的 **3+4** 算法。
  2. 找环主要是减少访存，尽量少使用数组，用**bool ， unsigned short** 来代替**int**会取得很好的效果。
  3. **if continue**结构特别慢，使用**while**结构代替**if continue**这种能够在1963W数据集上取得0.4秒的优化，主要思路是用**while**来维护左边界。
  4. 对于菊花图的处理，通过设置阈值**T** ,我们取值为30, 分菊花点和非菊花点进行处理，菊花点可以使用**二维数组**或者 **带压缩的一维数组** 进行访存，非菊花点可以使用 **vi * T+offset** 直接进行定位，通过一个**bool**数组进行区分,这一点让我们线上从5.X优化到了3.77。
  5. 使用**3+4**算法时利用插入排序代替**sort**,线下能够有0.2的收益。
  6. 使用 **while**结构代替 **for**结构，能够加速。
  7. 使用 **memcpy**每16字节对数组进行复制操作，能够实现加速，但是这也是**主要**让我们B榜失利的原因，官方改long long让我们这个复制memcpy结构在找环中失效了，过了一个多小时才发现这个问题。
  8. 为path3数组增加映射，假设本来从root走三步能到的点为 1,10,100,10000, 将其映射为 0,1,2,3; 达到一个节约空间的目的，防止B榜出现路径极多的结点把path3给爆了（担心是多余的，苦笑）；虽然多了一次访存，但是速度和不加映射相差无几，我们猜测是因为这样操作之后path3数组的下标变得连续了，对cache更加友好(减少换的次数)。
  9. 负载均衡方面：
    1） 我们最开始是将资源分配给结点，使用**伪**动态规划策略进行**负载均衡**，将结点每一段分配给一个线程进行处理。线上和线下都能够达到很好的效果，线上甚至和抢占式负载均衡策略相差无几，所以我认为这个负载均衡策略还是值得拿出来分享一下的，核心算法如下:
    ```
      dp[i][j]： 以i结点为root走j+1步能够到达的结点数量，这里保证每一步的下一个结点都要大于当前结点，所以这样得到的路径必然是升序，与题干要求不符。所以我们为了平衡这一点，让i结点多走了一步。从dp[i][0]计算到dp[i][6],累加前缀和，最后使用前缀和比例进行结点划分。
      初始化: if(u<v) dp[u][0]++;

      for(int i=nodeNum-1;i>=0;i--){
          int v;
          sum[i] = dp[i][0];
          for(us j=1;j<=6;j++){
              for(us k=0;k<ecnt[i];k++){
                  v = mp[i][k].v;
                  if(v<i)continue;
                  dp[i][j] += dp[v][j-1];
              }
              sum[i] += dp[i][j];
          }
      }
      for(int i=1;i<nodeNum;i++)sum[i] += sum[i-1];
      long long tmp = sum[nodeNum-1];
      printf("%lld\n",tmp);
      int pov1 = lower_bound(sum,sum+nodeNum,tmp*0.125) - sum;
      int pov2 = lower_bound(sum,sum+nodeNum,tmp*0.25) - sum;
      int pov3 = lower_bound(sum,sum+nodeNum,tmp*0.375) - sum;
      int pov4 = lower_bound(sum,sum+nodeNum,tmp*0.5) - sum;
      int pov5 = lower_bound(sum,sum+nodeNum,tmp*0.625) - sum;
      int pov6 = lower_bound(sum,sum+nodeNum,tmp*0.75) - sum;
      int pov7 = lower_bound(sum,sum+nodeNum,tmp*0.875) - sum;
      pool->pushJob(bind(mainWork,pov1+1,pov2,src2,1));
      pool->pushJob(bind(mainWork,pov2+1,pov3,src3,2));
      pool->pushJob(bind(mainWork,pov3+1,pov4,src4,3));
      pool->pushJob(bind(mainWork,pov4+1,pov5,src5,4));
      pool->pushJob(bind(mainWork,pov5+1,pov6,src6,5));
      pool->pushJob(bind(mainWork,pov6+1,pov7,src7,6));
      pool->pushJob(bind(mainWork,pov7+1,nodeNum-1,src8,7));
      mainWork(0,pov1,src1,0);
    ```
    2） 将资源分配给线程，将找环任务每10个进行划分放入任务队列，线程对资源对任务进行抢占，但是由于我们设计的架构问题，我们要对线程最终找到的环进行归并，这样会带来0.2的开销。

* 输出方面
  1. 首先我们使用归并的方式，对资源进行合并，改进了归并过程，由从小到大改为从大到小进行归并，省去了一个辅助数组的分配和复制。
  2. 在转和写方面，我们同样是让线程抢占式对任务进行转，我们通过设置**ok**标记，在转完一个任务之后马上就进行写入操作，达到了一个异步写入的效果。同时在**ok[i]!=true**的时候，让主线程也去任务队列拿任务做，参与转的过程，若**ok[i]==true**,则由主线程进行写入操作。转的过程中我们同样是使用了memcpy来对赋值进行加速。

* 参赛感想
  https://www.zhihu.com/question/395234431/answer/1227237191
