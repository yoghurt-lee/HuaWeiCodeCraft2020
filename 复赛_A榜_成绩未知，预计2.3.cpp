#include <iostream>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <bitset>
using namespace std;
struct Timer{
    timeval tic, toc;
    Timer(){
        gettimeofday(&tic,NULL);
    }

    void stop(const char* name)
    {
        gettimeofday(&toc,NULL);
        printf("%s: %f(s)\n", name, float(toc.tv_sec-tic.tv_sec) + float(toc.tv_usec-tic.tv_usec)/1000000);
    }
};

const int Threshold = 300;
const int N = 4000005, M = 2000003;
typedef unsigned char uc;
typedef short us;
typedef unsigned long long ull;
#define char uint8_t
unordered_map<ull,char>pid;
float cost_time[4];
///原始信息
struct rawEdge{
	int u,v,w;
	void operator = (const rawEdge &tmp){u=tmp.u;v=tmp.v;w=tmp.w;}
	bool operator <(const rawEdge &tmp)const{
		return u==tmp.u ? (v<tmp.v) : (u<tmp.u);
	}
    bool operator <=(const rawEdge& b) const{return !(b < *this);}
    bool operator >=(const rawEdge& b) const{return !(*this < b);}
}*rawedge;

struct Edge{
    int v,w;
};

struct NUM{
    char p[15],len;
}Num[N];

void transNum(NUM *num,int x){
	char tmp[15],k = 0;
	char *p = num->p;
	int i=0,j;
	do{
		j = x/10;
		tmp[i++] = x-j*10+48;
		x = j;
	}while(x);
	for(int l=i-1;l>=0;l--)p[k++] = tmp[l];
	p[k++] = ',';
	p[k] = '\0';
	num->len = k;
}
///多线程相关
class ThreadPool{ /// 线程池
public:
	using JobFunc = function<void()>; /// 线程要执行的函数

	ThreadPool(int threadNum);
	~ThreadPool();
	void pushJob(const JobFunc &job);

public:
	vector<thread> threads_; /// 线程池
	JobFunc jobs_[N+500]; /// 任务队列
	int rear,head;
	condition_variable cond_; /// 条件变量 不满足的时候 会解锁+挂起等待  唤醒时自动加锁
	std::mutex lock_,worklock_;
	int working_,threadNum,cnt; /// 是否处于工作状态
	bool stop_; /// 析构时使用 销毁线程池
};
ThreadPool * pool;
ThreadPool::ThreadPool(int threadNum):working_(0),threadNum(threadNum),stop_(false),rear(0),head(0),cnt(0){
	for(char i=0;i<threadNum;i++){
		threads_.emplace_back([&](){
            {
                unique_lock<mutex> lock(lock_);
                pid[pthread_self()] = ++cnt;
            }
			while(true){
				JobFunc job;
				{
					unique_lock<mutex> lock(lock_);
					while(!stop_&&rear==head) cond_.wait(lock);
					if(stop_)return;
					job = jobs_[head++];
				}
				{
				    unique_lock<mutex> lock(worklock_);
				    working_++;
				}
				if(job!=nullptr)job();
				{
				    unique_lock<mutex> lock(worklock_);
				    working_--;
				}
			}
		});
	}
}

ThreadPool::~ThreadPool(){
	stop_=1;
	cond_.notify_all();
	for(auto &ths : threads_)ths.join(); /// 销毁所有线程
}

void ThreadPool::pushJob(const JobFunc &job){
	{
		unique_lock<mutex> lock(lock_);
		jobs_[rear++] = job;
	}
	cond_.notify_one();
}

void handon(){
    function<void()> func;
	while(1){
        {
            unique_lock<mutex> lock(pool->lock_);
            if(pool->rear==pool->head)break;
            func = pool->jobs_[pool->head++];
        }
        func();
	}
}

///图信息
int nodeNum,edgeNum;
us indegree[N],outdegree[N];

us max_outd, max_ind; /// 所有点走一步的最大出度
us max_outd1 , max_ind1;
bool *out_ok,*in_ok;
///使用动态内存代替前向星
Edge *mp;
Edge *mp_;

Edge *mp1;
Edge *mp1_;
bool worth[N],worth1[N];
us ecnt[N],ecnt_[N];
char secnt[N],secnt_[N];

int cnt_in[N],cnt_out[N];
int sum_in[N],sum_out[N];
int edgepatition(int l,int r){
    int pos = (l+r)/2;
    rawEdge p = rawedge[pos];
    swap(rawedge[l],rawedge[pos]);
    while(l<r){
        while(l<r && rawedge[r]>=p) r--;
        rawedge[l] = rawedge[r];
        while(l<r && rawedge[l]<=p) l++;
        rawedge[r] = rawedge[l];
    }
    rawedge[l] = p;
    return l;
}
void packedgesort(const int &l,const int &r){sort(rawedge+l,rawedge+r);}

void myedgeSort(const int &l,const int &r){
    if(l>=r)return;
	int pov = edgepatition(l,r);
	if(pov-l<=40000)pool->pushJob(bind(&packedgesort,l,pov));
	else myedgeSort(l,pov-1);
	if(r-pov<=40000)pool->pushJob(bind(&packedgesort,pov+1,r+1));
	else myedgeSort(pov+1,r);
}

///找环相关
struct PATH3{
    int v_,v,w_,w;
    void operator = (const PATH3 &tmp){v=tmp.v;v_=tmp.v_;w_=tmp.w_;w=tmp.w;}
    bool operator <(const PATH3 &tmp)const{
		return v_==tmp.v_ ? (v<tmp.v) : (v_<tmp.v_);
	}
};

struct Source{
    bool *vis1,*vis2;
    char *pcnt3;
    us *nodeys;
    int *ans_3,*ans_4,*ans_5,*ans_6,*ans_7;
    int key3,key4,key5,key6,key7;
    int *hasModify_3;

    int modify_cnt3;
    PATH3 *path_3;
}*src[4];

void positive(const int &root, Source *src){
	int v[8];
	long long w[5],wx3,wx5;
	v[0] = root;
	int &key3 = src->key3;
	int &key4 = src->key4;
	int &key5 = src->key5;
	int &key6 = src->key6;
	int &key7 = src->key7;
	int *ans_3 = src->ans_3,*ans_4 = src->ans_4;
	int *ans_5 = src->ans_5,*ans_6 = src->ans_6, *ans_7= src->ans_7;
    char *pcnt3 = src->pcnt3;
	PATH3 *path_3 = src->path_3;
    bool *vis1 = src->vis1;
    bool *vis2 = src->vis2;
    us *nodeys = src->nodeys;
    Edge * E1;
    us k = 0, r;
    if(out_ok[v[0]]){
        E1 = mp1+v[0]*30;
        r = secnt[v[0]];
    }else{
        E1 = mp+sum_out[v[0]];
        r = ecnt[v[0]];
    }
    while(k<r && E1->v<=root) E1++,k++;
    while(k++<r){
        v[1] = (E1)->v, w[0] = (E1++)->w;
	    wx3 = w[0]*3; wx5 = w[0]*5;
		vis2[v[1]] = 1;
		if(vis1[v[1]]){
		    PATH3 *path = path_3+nodeys[v[1]]*Threshold;
            char k = 0,r = pcnt3[v[1]];
            for(char i=1;i<r;i++){
                short j=i;
                while(j && path[j]<path[j-1]){
                    swap(path[j],path[--j]);
                }
            }
            while(k++<r){
                int t[4];
                memcpy(t,path++,16);
                if(!vis2[t[1]] &&!vis2[t[0]]
                   &&w[0]<=5ll*t[2] && 3*w[0]>=t[2]
                   && t[3]<=5*w[0] && 3ll*t[3]>=w[0])
                {
                    memcpy(ans_4+key4,v,8);
                    memcpy(ans_4+key4+2,t,8);
                    //ans_4[key4+2] = t[1];
                    //ans_4[key4+3] = t[0];
                    key4+=4;
                }
            }
        }
        Edge * E2;
        us k = 0, r;
        if(out_ok[v[1]]){
            E2 = mp1+v[1]*30;
            r = secnt[v[1]];
        }else{
            E2 = mp+sum_out[v[1]];
            r = ecnt[v[1]];
        }
        while(k<r && E2->v<=root) E2++,k++;
        while(k++<r){
            v[2] = (E2)->v, w[1] = (E2++)->w;
			if(vis2[v[2]] || wx3<w[1] || w[0]>5*w[1])continue;
            vis2[v[2]] = 1;
			if(vis1[v[2]]){
                PATH3 *path = path_3+nodeys[v[2]]*Threshold;
                char k = 0,r = pcnt3[v[2]];
                for(char i=1;i<r;i++){
                    short j=i;
                    while(j && path[j]<path[j-1]){
                        swap(path[j],path[--j]);
                    }
                }
                while(k++<r){
                    int t[4];
                    memcpy(t,path++,16);
                    if(!vis2[t[1]] &&!vis2[t[0]]
                       &&w[1]<=5ll*t[2] && 3*w[1]>=t[2]
                       && t[3]<=5*w[0] && 3ll*t[3]>=w[0])
                    {
                        memcpy(ans_5+key5,v,12);
                        memcpy(ans_5+key5+3,t,8);
                        //ans_5[key5+3] = t[1];
                        //ans_5[key5+4] = t[0];
                        key5 += 5;
                    }
                }
            }
			Edge *E3;
            us k = 0 ,r;
			if(out_ok[v[2]]){
                E3 = mp1+v[2]*30;
                r = secnt[v[2]];
            }else{
                E3 = mp+sum_out[v[2]];
                r = ecnt[v[2]];
			}

            while(k<r && E3->v<root) E3++,k++;
            while(k++<r){
                v[3] = (E3)->v, w[2] = (E3++)->w;
                if(3*w[1]<w[2] || w[1]>5*w[2]) continue;
                if(v[3] == root){
                    if(3*w[2]>=w[0] && w[2]<=w[0]*5){
                        memcpy(ans_3+key3, v, 16);
                        key3+=3;
                    }
                    continue;
                }
                if(vis2[v[3]])continue;

                if(vis1[v[3]]){
                    PATH3 *path = path_3+nodeys[v[3]]*Threshold;
                    us k = 0,r = pcnt3[v[3]];

                    for(us i=1;i<r;i++){
                        short j=i;
                        while(j && path[j]<path[j-1]){
                            swap(path[j],path[--j]);
                        }
                    }
                    while(k++<r){
                        int t[4];
                        memcpy(t,path++,16);
                        if(!vis2[t[1]] &&!vis2[t[0]]
                           &&w[2]<=5ll*t[2] && 3*w[2]>=t[2]
                           && t[3]<=5*w[0] && 3ll*t[3]>=w[0])
                        {
                            memcpy(ans_6+key6,v,16);
                            memcpy(ans_6+key6+4,t,8);
                            //ans_6[key6+4] = t0;
                            //ans_6[key6+5] = t1;
                            key6 += 6;
                        }
                    }
                }
                /// 2+4
                vis2[v[3]] = 1;
                us k = 0 ,r;
                Edge * E4;
                if(out_ok[v[3]]){
                    E4 = mp1+v[3]*30;
                    r = secnt[v[3]];
                }else{
                    E4 = mp+sum_out[v[3]];
                    r = ecnt[v[3]];
                }
                while(k<r && E4->v<=root) E4++,k++;
                while(k++<r){
                    v[4] = (E4)->v, w[3] = (E4++)->w;
                    if(!vis1[v[4]]||vis2[v[4]]||3*w[2]<w[3]||w[2]>5*w[3])continue;
                    /// 2+5
                    PATH3 *path = path_3+nodeys[v[4]]*Threshold;
                    us k = 0,r = pcnt3[v[4]];
                    for(us i=1;i<r;i++){
                        short j=i;
                        while(j && path[j]<path[j-1]){
                            swap(path[j],path[--j]);
                        }
                    }
                    while(k++<r){
                        int t[4];
                        memcpy(t,path++,16);
                        if(!vis2[t[1]] &&!vis2[t[0]]
                           &&w[3]<=5ll*t[2] && 3*w[3]>=t[2]
                           && t[3]<=5*w[0] && 3ll*t[3]>=w[0])
                        {
                            memcpy(ans_7+key7,v,20);
                            memcpy(ans_7+key7+5,t,8);
                            //ans_7[key7+5] = t0;
                            //ans_7[key7+6] = t1;
                            key7 += 7;
                        }
                    }
                }vis2[v[3]] = 0;
			}vis2[v[2]]=0;
		}vis2[v[1]] = 0;
	}vis2[v[0]] = 0;
}

void negative(const int &root, const int &root1 ,const int &_w,Source *src){
    int v[3], w[2];
	v[0] = root1;
	int *hasModify_3 = src->hasModify_3;
	int &modify_cnt3 = src->modify_cnt3;
	char *pcnt3 = src->pcnt3;
	PATH3 *path_3 = src->path_3;
	us *nodeys = src->nodeys;
	bool *vis1 = src->vis1;
	Edge *tmp;
	us k = 0, r;
	if(in_ok[v[0]]){
        tmp = mp1_+v[0]*30;
        r = secnt_[v[0]];
	}else{
        tmp = mp_+sum_in[v[0]];
        r = ecnt_[v[0]];
	}
    while(k<r && tmp->v<=root) tmp++,k++;
	while(k++<r){
		v[1] = tmp->v, w[0] = (tmp++)->w;
		if(w[0]>5ll*_w || 3ll*w[0]<_w ) continue;
        Edge *tmp1;
        if(in_ok[v[1]]){
            tmp1 = mp1_+v[1]*30;
            char k = 0, r = secnt_[v[1]];
            while(k<r && tmp1->v<=root) tmp1++,k++;
            while(k++<r){
                v[2] = tmp1->v, w[1] = (tmp1++)->w;
                if(v[0]==v[2] || w[1]>5ll*w[0] || 3ll*w[1]<w[0]) continue;
                if(!vis1[v[2]]){
                    nodeys[v[2]] = modify_cnt3;
                    hasModify_3[modify_cnt3++] = v[2];
                    vis1[v[2]]=1;
                }
                *(path_3+nodeys[v[2]]*Threshold+pcnt3[v[2]]++) = {v[1],v[0],w[1],_w};
            }
        }else{
            tmp1 = mp_+sum_in[v[1]];
            us k = 0, r = ecnt_[v[1]];
            while(k<r && tmp1->v<=root) tmp1++,k++;
            while(k++<r){
                v[2] = tmp1->v, w[1] = (tmp1++)->w;
                if(v[0]==v[2] || w[1]>5ll*w[0] || 3ll*w[1]<w[0]) continue;
                if(!vis1[v[2]]){
                    nodeys[v[2]] = modify_cnt3;
                    hasModify_3[modify_cnt3++] = v[2];
                    vis1[v[2]]=1;
                }
                *(path_3+nodeys[v[2]]*Threshold+pcnt3[v[2]]++) = {v[1],v[0],w[1],_w};
            }
        }
	}
}

void mainWork(const int &l,const int &r){
    Timer t;
    int sid = pid[pthread_self()];
    ///找环
    Source *src_ = src[sid];
    bool *vis1 = src_->vis1;
    PATH3 *path_3 = src_->path_3;
    char *pcnt3 = src_->pcnt3;
    us *nodeys = src_->nodeys;
    for(int root=l;root<=r;root++){
        if(!worth[root]||!worth1[root])continue;

        for(us i=src_->modify_cnt3-1;i>=0;i--){
            int a = src_->hasModify_3[i];
            pcnt3[a] = vis1[a] = nodeys[a]= 0;
        }

        src_->modify_cnt3 = 0;
        if(in_ok[root]){
            Edge *tmp = mp1_+root*30;
            char k = 0, r = secnt_[root];
            while(k<r && tmp->v<=root) tmp++,k++;
            while(k++<r){
                negative(root, tmp->v, tmp->w, src_);
                tmp++;
            }
        }else{
            Edge *tmp = mp_+sum_in[root];
            us k = 0, r = ecnt_[root];
            while(k<r && tmp->v<=root) tmp++,k++;
            while(k++<r){
                negative(root, tmp->v, tmp->w, src_);
                tmp++;
            }
        }
        positive(root, src_);
    }
    gettimeofday(&t.toc,NULL);
    cost_time[sid] += float(t.toc.tv_sec-t.tic.tv_sec) + float(t.toc.tv_usec-t.tic.tv_usec)/1000000;
}

void allocResource(Source *src){
    src->vis1 = (bool *)malloc(sizeof(bool)*nodeNum);
    src->vis2 = (bool *)malloc(sizeof(bool)*nodeNum);
    src->pcnt3 = (char *) malloc(nodeNum*sizeof(char));
    src->nodeys = (us *)malloc(sizeof(us)*nodeNum);
    src->ans_3 = (int *) malloc(3*20000000*sizeof(int));
    src->ans_4 = (int *) malloc(4*20000000*sizeof(int));
    src->ans_5 = (int *) malloc(5*20000000*sizeof(int));
    src->ans_6 = (int *) malloc(6*20000000*sizeof(int));
    src->ans_7 = (int *) malloc(7*20000000*sizeof(int));
    src->hasModify_3 = (int *) malloc(nodeNum*sizeof(int));

    src->path_3 = (PATH3 *)malloc(sizeof(PATH3)*Threshold*50000);
}

void findCircle() {
    pid[pthread_self()] = 0;
    for(int i=0;i<nodeNum;i+=10){
        pool->pushJob(bind(mainWork,i,min(i+9,nodeNum-1)));
    }

    handon();
    while(pool->working_>0)usleep(100);
}

///IO相关
int A[M<<1];
int patition(int l,int r){
    int pos = (l+r)>>1;
    int p = A[pos];
    swap(A[l],A[pos]);
    while(l<r){
        while(l<r && A[r]>=p) r--;
        A[l] = A[r];
        while(l<r && A[l]<=p) l++;
        A[r] = A[l];
    }
    A[l] = p;
    return l;
}
void packsort(const int &l,const int &r){sort(A+l,A+r);}
void mySort(const int &l,const int &r){
    if(l>=r)return;
	int pov = patition(l,r);
	if(pov-l<=10000)pool->pushJob(bind(&packsort,l,pov));
	else mySort(l,pov-1);
	if(r-pov<=10000)pool->pushJob(bind(&packsort,pov+1,r+1));
	else mySort(pov+1,r);
}

const int top = 1e9;
int *ys;
bool *hs;

void mytransfer(int i,const int &j){
    for(;i<=j;i++){
        transNum(&Num[i],A[i]);
         if(A[i]<top)ys[A[i]]=i;
    }
}
void mylowerbound(int i,const int &j){
    for(;i<=j;i++){
        int u = rawedge[i].u,v = rawedge[i].v;
        if(u<top)rawedge[i].u = ys[u];
        else rawedge[i].u = lower_bound(A,A+nodeNum, u)-A;
        if(v<top)rawedge[i].v = ys[v];
        else rawedge[i].v = lower_bound(A,A+nodeNum, v)-A;
    }
}

void readAndDraw(){
	//int fd = open("/data/test_data.txt",O_RDONLY,0664);
     int fd = open("test_data_19630345.txt",O_RDONLY,0664);
    //int fd = open("test_data_18159792__.txt",O_RDONLY,0664);
    // int fd = open("test_data_1305172.txt",O_RDONLY,0664);
   // int fd = open("test_data_18875018.txt",O_RDONLY,0664);
	int siz = lseek(fd,0,SEEK_END);
	char *p = (char *)mmap(NULL,siz,PROT_READ,MAP_PRIVATE,fd,0);
	int u=0,v=0,w = 0,index=0;
	char c,comma=0;
	Timer t = Timer();
	hs = (bool *)malloc(sizeof(bool)*top);
	rawedge = (rawEdge *)malloc(sizeof(rawEdge)*M);
	while(index<siz){
        while((c = p[index++]) >= 48){
            if(comma == 0)u = u*10+c-'0';
            else if(comma == 1) v = v*10+c-'0';
            else w = w*10 + c-'0';
        }
        if(c == 44)comma++;
        else{
            if(u!=v){
                if(u>top||!hs[u])u<=top?(hs[u]=1):(1),A[nodeNum++] = u;
                if(v>top||!hs[v])v<=top?(hs[v]=1):(1),A[nodeNum++] = v;
                rawedge[edgeNum++] = {u,v,w};
            }
            u = v = w = comma = 0;
        }
	}
	t.stop("reading");

    t = Timer();
    mySort(0,nodeNum-1);
    handon();
    while(pool->working_>0)usleep(100);
    nodeNum = unique(A,A+nodeNum)-A;
    t.stop("sort and unique node");
    t = Timer();
    ys = (int *)malloc(sizeof(int)*top);
	for(int i=0;i<nodeNum;i+=5000){
        pool->pushJob(bind(&mytransfer,i,min(nodeNum-1,i+4999)));
	}
	handon();
    while(pool->working_>0)usleep(100);
    in_ok = (bool *)malloc(sizeof(bool)*nodeNum);
    out_ok = (bool *)malloc(sizeof(bool)*nodeNum);
    t.stop("transfer number");

    t = Timer();
    for(int i=0;i<edgeNum;i+=5000) {
        pool->pushJob(bind(&mylowerbound,i, min(edgeNum-1,i+4999)));
	}
    handon();
    while(pool->working_>0)usleep(100);
    t.stop("discrete number");

    t = Timer();
    myedgeSort(0,edgeNum-1);
	handon();
	while(pool->working_>0)usleep(100);
    t.stop("sort edge");

    t = Timer();
	for(int i=0;i<edgeNum;i++){++indegree[rawedge[i].v]; ++outdegree[rawedge[i].u];}
    max_ind1 = max_outd1 = 30;
    for(int i=0;i<nodeNum;i++){
        sum_in[i+1] = sum_in[i];
        sum_out[i+1] = sum_out[i];
        if(outdegree[i]<=max_outd1) out_ok[i] = 1;
        else sum_out[i+1] += outdegree[i];
        if(indegree[i]<=max_ind1) in_ok[i] = 1;
        else sum_in[i+1] += indegree[i];
    }

    for(int i=0;i<4;i++){
        src[i] = (Source *) malloc(sizeof(Source));
        pool->pushJob(bind(allocResource,src[i]));
    }

    mp = (Edge *)malloc(sizeof(Edge)*sum_out[nodeNum]);
    mp_ = (Edge *)malloc(sizeof(Edge)*sum_in[nodeNum]);

    mp1 = (Edge *)malloc(sizeof(Edge)*nodeNum*30);
    mp1_ = (Edge *)malloc(sizeof(Edge)*nodeNum*30);
    int i=0;
	while(i++<edgeNum){
		u = rawedge->u, v = (rawedge)->v , w = (rawedge++)->w;
		if(indegree[u]&&outdegree[v]){
			if(u<v)worth[u] = 1;
			else worth1[v] = 1;
            if(out_ok[u]){
                mp1[u*30+secnt[u]++ ] = {v,w};
            }
            else {
                mp[sum_out[u]+ecnt[u]++] = {v,w};
            }
			if(in_ok[v]) mp1_[v*max_ind1+secnt_[v]++ ] = {u,w};
			else mp_[sum_in[v]+ecnt_[v]++] = {u,w};
		}
	}


	handon();
	while(pool->working_>0)usleep(100);
	t.stop("build");
}

char *s;
int len[101],seg[7][2],allblock,length;
bool ok[101];
const int block_length = 16000000;

void store3(const int &l,const int &r,const int &id,int &k){
    int *ans_3 = src[0]->ans_3;
    int pos = l*3;
    for(int i=l;i<=r;i++){
        for(char j=0;j<3;j++){
            memcpy(s+id*block_length+k,Num[ans_3[pos]].p,16);
            k+=Num[ans_3[pos++]].len;
        }
        k--;   *(s+id*block_length+k++) = '\n';
    }
}
void store4(const int &l,const int &r,const int &id,int &k){
    int *ans_4 = src[0]->ans_4;
    int pos = l*4;
    for(int i=l;i<=r;i++){
        for(char j=0;j<4;j++){
            memcpy(s+id*block_length+k,Num[ans_4[pos]].p,16);
            k+=Num[ans_4[pos++]].len;
        }
        k--;   *(s+id*block_length+k++) = '\n';
    }
}
void store5(const int &l,const int &r,const int &id,int &k){
    int *ans_5 = src[0]->ans_5;
    int pos = l*5;
    for(int i=l;i<=r;i++){
        for(char j=0;j<5;j++){
            memcpy(s+id*block_length+k,Num[ans_5[pos]].p,16);
            k+=Num[ans_5[pos++]].len;
        }
        k--;   *(s+id*block_length+k++) = '\n';
    }
}
void store6(const int &l,const int &r,const int &id,int &k){
    int *ans_6 = src[0]->ans_6;
    int pos = l*6;
    for(int i=l;i<=r;i++){
        for(int j=0;j<6;j++){
            memcpy(s+id*block_length+k,Num[ans_6[pos]].p,16);
            k+=Num[ans_6[pos++]].len;
        }
        k--;   *(s+id*block_length+k++) = '\n';
    }
}
void store7(const int &l,const int &r,const int &id,int &k){
    int *ans_7 = src[0]->ans_7;
    int pos = l*7;
    for(int i=l;i<=r;i++){
        for(char j=0;j<7;j++){
            memcpy(s+id*block_length+k,Num[ans_7[pos]].p,16);
            k+=Num[ans_7[pos++]].len;
        }
        k--;   *(s+id*block_length+k++) = '\n';
    }
}

void transfer(const int &l,const int &r,const int &id){
    int a,b,k=0;
    for(char i=0;i<5;i++){
        if(seg[i][0]>r||seg[i][1]<l)continue;
        a = max(seg[i][0],l) - seg[i][0];
        b = min(seg[i][1],r) - seg[i][0];
        switch(i){
            case 4: store7(a,b,id,k); break;
            case 3: store6(a,b,id,k); break;
            case 2: store5(a,b,id,k); break;
            case 1: store4(a,b,id,k); break;
            default: store3(a,b,id,k); break;
        }
    }
    ok[id] = 1;
    len[id] = k;
}

bool cmp(int *a,int *b,const int &ty){
    char i = 0;
    while(i<ty){
        if(*a != *b)return *a < *b;
        a++;b++;
    }
    return 1;
}

void merge_(int *ans1,const int &a,int *ans2,const int &b,const int &ty){
    int tot = a+b,i=a-ty,j=b-ty,now=a+b-ty;
    while(i>=0&&j>=0){
        if(cmp(ans1+i,ans2+j,ty)){
            memcpy(ans1+now,ans2+j,ty*4);
            j-=ty;now-=ty;
        }else{
            memcpy(ans1+now,ans1+i,ty*4);
            i-=ty;now-=ty;
        }
    }
    while(j>=0){
        memcpy(ans1+now,ans2+j,ty*4);
        j-=ty;now-=ty;
    }
}

void getans3(){
    int a = src[0]->key3,b = src[1]->key3;
    int c = src[2]->key3,d = src[3]->key3;
    int *ans1 = src[0]->ans_3,*ans2 = src[1]->ans_3;
    int *ans3 = src[2]->ans_3,*ans4 = src[3]->ans_3;
    pool->pushJob(bind(merge_,ans1,a,ans2,b,3));
    pool->pushJob(bind(merge_,ans3,c,ans4,d,3));
    src[0]->key3 = a+b;
    src[2]->key3 = c+d;
}
void getans4(){
    int a = src[0]->key4,b = src[1]->key4;
    int c = src[2]->key4,d = src[3]->key4;
    int *ans1 = src[0]->ans_4,*ans2 = src[1]->ans_4;
    int *ans3 = src[2]->ans_4,*ans4 = src[3]->ans_4;
    pool->pushJob(bind(merge_,ans1,a,ans2,b,4));
    pool->pushJob(bind(merge_,ans3,c,ans4,d,4));
    src[0]->key4 = a+b;
    src[2]->key4 = c+d;
}
void getans5(){
    int a = src[0]->key5,b = src[1]->key5;
    int c = src[2]->key5,d = src[3]->key5;
    int *ans1 = src[0]->ans_5,*ans2 = src[1]->ans_5;
    int *ans3 = src[2]->ans_5,*ans4 = src[3]->ans_5;
    pool->pushJob(bind(merge_,ans1,a,ans2,b,5));
    pool->pushJob(bind(merge_,ans3,c,ans4,d,5));
    src[0]->key5 = a+b;
    src[2]->key5 = c+d;
}

void getans6(){
    int a = src[0]->key6,b = src[1]->key6;
    int c = src[2]->key6,d = src[3]->key6;
    int *ans1 = src[0]->ans_6,*ans2 = src[1]->ans_6;
    int *ans3 = src[2]->ans_6,*ans4 = src[3]->ans_6;
    pool->pushJob(bind(merge_,ans1,a,ans2,b,6));
    pool->pushJob(bind(merge_,ans3,c,ans4,d,6));
    src[0]->key6 = a+b;
    src[2]->key6 = c+d;
}

void getans7(){
    int a = src[0]->key7,b = src[1]->key7;
    int c = src[2]->key7,d = src[3]->key7;
    int *ans1 = src[0]->ans_7,*ans2 = src[1]->ans_7;
    int *ans3 = src[2]->ans_7,*ans4 = src[3]->ans_7;
    pool->pushJob(bind(merge_,ans1,a,ans2,b,7));
    pool->pushJob(bind(merge_,ans3,c,ans4,d,7));
    src[0]->key7 = a+b;
    src[2]->key7 = c+d;
}

void myhandon(int num){
    function<void()> func;
    while(num--){
        {
            unique_lock<mutex> lock(pool->lock_);
            if(pool->rear==pool->head)return;
            func = pool->jobs_[pool->head++];
        }
        func();
    }
}

void save(int fd){
    int ret,tot = src[0]->key3/3+src[0]->key4/4+src[0]->key5/5+src[0]->key6/6+src[0]->key7/7;
    int ls = 0,k=0,now = tot,nx;
    char tmp[20],p[20];
    do{
        nx = now/10;
        tmp[ls++] = now-nx*10+48;
        now=nx;
    }while(now);
    length += ls;
    for(int l=ls-1;l>=0;l--) p[k++] = tmp[l];
    p[k++] = '\n';
    write(fd,p,k);
    for(int i=0;i<allblock;i++){
        while(!ok[i])myhandon(1);
        write(fd,s+i*block_length,len[i]);
    }
}

int main(){
	pool = new ThreadPool(3);
    Timer T;
	readAndDraw();
	T.stop("loading time: ");

	printf("node num:%d\n",nodeNum);
	printf("edge num:%d\n",edgeNum);

    T = Timer();
	findCircle();
	for(int i=0;i<4;i++)printf("%d pthread cost : %f(s)\n",i,cost_time[i]);
    T.stop("circle time: ");

    T = Timer();
    pool->pushJob(bind(getans7));
    pool->pushJob(bind(getans6));
    pool->pushJob(bind(getans5));
    pool->pushJob(bind(getans4));
    pool->pushJob(bind(getans3));
    handon();
    while(pool->working_>0)usleep(100);
    pool->pushJob(bind(merge_,src[0]->ans_6,src[0]->key6,src[2]->ans_6,src[2]->key6,6));
    pool->pushJob(bind(merge_,src[0]->ans_5,src[0]->key5,src[2]->ans_5,src[2]->key5,5));
    pool->pushJob(bind(merge_,src[0]->ans_4,src[0]->key4,src[2]->ans_4,src[2]->key4,4));
    pool->pushJob(bind(merge_,src[0]->ans_3,src[0]->key3,src[2]->ans_3,src[2]->key3,3));
    merge_(src[0]->ans_7,src[0]->key7,src[2]->ans_7,src[2]->key7,7);
    handon();
    while(pool->working_>0)usleep(100);
    src[0]->key3 += src[2]->key3;
    src[0]->key4 += src[2]->key4;
    src[0]->key5 += src[2]->key5;
    src[0]->key6 += src[2]->key6;
    src[0]->key7 += src[2]->key7;
    T.stop("merge ans");

    int key_3=src[0]->key3/3,key_4=src[0]->key4/4,key_5=src[0]->key5/5,key_6=src[0]->key6/6,key_7=src[0]->key7/7;
    int tot = key_3+key_4+key_5+key_6+key_7;
    printf("%d %d %d %d %d\n",key_3,key_4,key_5,key_6,key_7);

    seg[0][0] = 0,seg[0][1]=key_3-1;
    seg[1][0] = key_3,seg[1][1]=key_3+key_4-1;
    seg[2][0] = key_3+key_4,seg[2][1]=key_3+key_4+key_5-1;
    seg[3][0] = key_3+key_4+key_5,seg[3][1]=key_3+key_4+key_5+key_6-1;
    seg[4][0] = key_3+key_4+key_5+key_6,seg[4][1]=key_3+key_4+key_5+key_6+key_7-1;

    s = (char *) malloc(100 * block_length * sizeof(char));

    T = Timer();
    int beg_ =0,end_=-1;
    allblock = 0;

    while(tot>0){
        beg_ = end_+1;
        if(tot>200000){
            end_ += 200000;
            tot -= 200000;
        }else{
            end_ += tot;
            tot = 0;
        }
        pool->pushJob(bind(&transfer,beg_,end_,allblock));
        allblock++;
    }
    //handon();
    //while(pool->working_>0)sleep(0.0001);
    int fd = open("/projects/student/result.txt",O_CREAT|O_RDWR,0666);
    save(fd);
    T.stop("saving time: ");
    printf("circle num: %d\n",key_3+key_4+key_5+key_6+key_7);
}
