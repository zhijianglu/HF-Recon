//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_LC_CALIB_UTILS_H
#define CHECKERBOARD_LC_CALIB_UTILS_H
#include <sys/stat.h>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include "getfile.h"
#include<experimental/filesystem>


inline int rm_dir(std::string dir_full_path) {
    DIR* dirp = opendir(dir_full_path.c_str());
    if(!dirp)
    {
        return -1;
    }
    struct dirent *dir;
    struct stat st;
    while((dir = readdir(dirp)) != NULL)
    {
        if(strcmp(dir->d_name,".") == 0
            || strcmp(dir->d_name,"..") == 0)
        {
            continue;
        }
        std::string sub_path = dir_full_path + '/' + dir->d_name;
        if(lstat(sub_path.c_str(),&st) == -1)
        {
            //Log("rm_dir:lstat ",sub_path," error");
            continue;
        }
        if(S_ISDIR(st.st_mode))
        {
            if(rm_dir(sub_path) == -1) // 如果是目录文件，递归删除
            {
                closedir(dirp);
                return -1;
            }
            rmdir(sub_path.c_str());
        }
        else if(S_ISREG(st.st_mode))
        {
            unlink(sub_path.c_str());     // 如果是普通文件，则unlink
        }
        else
        {
            //Log("rm_dir:st_mode ",sub_path," error");
            continue;
        }
    }
    if(rmdir(dir_full_path.c_str()) == -1)//delete dir itself.
    {
        closedir(dirp);
        return -1;
    }
    closedir(dirp);
    return 0;
}

inline bool Remove(std::string file_name) {
    std::string file_path = file_name;
    struct stat st;
    if (lstat(file_path.c_str(),&st) == -1) {
        return EXIT_FAILURE;
    }
    if (S_ISREG(st.st_mode)) {
        if (unlink(file_path.c_str()) == -1) {
            return EXIT_FAILURE;
        }
    }
    else if(S_ISDIR(st.st_mode)) {
        if(file_name == "." || file_name == "..") {
            return EXIT_FAILURE;
        }
        if(rm_dir(file_path) == -1)//delete all the files in dir.
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

static
void create_dir(std::string dirName_in, bool remove_before = true)
{

    if(!opendir(dirName_in.c_str()))  //如果不存在那就直接创建
    {
        mkdir(dirName_in.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);  //创建输出深度图文件夹
    }else if(remove_before){
        Remove(dirName_in);//删除之前的文件
        mkdir(dirName_in.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);  //创建输出深度图文件夹
    }
}

static bool is_in_list(std::string str, std::vector<std::string> list){
    for(auto &s: list){
        if(str.find(s) != std::string::npos){
            return true;
        }
    }
    return false;
}

static  int get_file_list (std::string dir, std::vector<std::string> &files, bool dir_or_file, std::vector<std::string> filted_dirs=std::vector<std::string>{})  //file true, dir false
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    std::vector<std::string> file_names;

    while ((dirp = readdir(dp)) != NULL) {
//    while ((dirp = scandir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);
        if(name != "." && name != ".." )
        {
            //std::cout<<name<<std::endl;
            file_names.push_back(name);
        }
    }
    closedir(dp);


    std::sort(file_names.begin(), file_names.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
    for(unsigned int i=0;i<file_names.size();i++)
    {
        if(file_names[i].at(0) != '/')
        {
            if(is_in_list(file_names[i], filted_dirs)){
                continue;
            }

            std::string file_path = dir + file_names[i];
            if(std::experimental::filesystem::is_directory(file_path) ^ dir_or_file){
                files.push_back(file_path);
            }
        }
    }

    return files.size();
}


static
void
load_file_path(std::string file_path, std::vector<std::string> &vstr_file_path, bool sort_ = true)
{

    if (getdir(file_path, vstr_file_path) >= 0)
    {
//        printf("found %d files in folder %s!\n",
//               (int) vstr_file_path.size(),
//               file_path_down.c_str());
    }
    else if (getFile(file_path.c_str(), vstr_file_path) >= 0)
    {
//        printf("found %d files in file %s!\n",
//               (int) vstr_file_path.size(),
//               file_path_down.c_str());
    }
    else
    {
        //LOG(INFO)<<"path: "<<file_path;
        printf("could not load file list! wrong path / file?\n");
        return;
    }
    if (!sort_)
    {
        return;
    }

    sort(vstr_file_path.begin(), vstr_file_path.end(), [](std::string x, std::string y)
         {

             int p1 = x.find_last_of("/");
             std::string sub_str = x.substr(p1 + 1, x.size() - p1);
             int a_t1 = atoi(sub_str.substr(0, sub_str.find_first_of("-")).c_str());
             int a_t2 = atoi(sub_str.substr(sub_str.find_first_of("-") + 1,
                                            sub_str.find_last_of("-") - sub_str.find_first_of("-") - 1).c_str());
             int a_t3 =
                 atoi(sub_str.substr(sub_str.find_last_of("-") + 1, sub_str.size() - sub_str.find_last_of("-")).c_str());

             p1 = y.find_last_of("/");
             sub_str = y.substr(p1 + 1, y.size() - p1);
             int b_t1 = atoi(sub_str.substr(0, sub_str.find_first_of("-")).c_str());
             int b_t2 = atoi(sub_str.substr(sub_str.find_first_of("-") + 1,
                                            sub_str.find_last_of("-") - sub_str.find_first_of("-") - 1).c_str());
             int b_t3 =
                 atoi(sub_str.substr(sub_str.find_last_of("-") + 1, sub_str.size() - sub_str.find_last_of("-")).c_str());


//             string s_time = x
//                 .substr(x.find_last_of("/") + 1, x.find_last_of(".") - x.find_last_of("/") - 1);
//             double a_stamp = atof(s_time.c_str());
//
//             s_time = y
//                 .substr(y.find_last_of("/") + 1, y.find_last_of(".") - y.find_last_of("/") - 1);
//             double b_stamp = atof(s_time.c_str());
             if (a_t1 > b_t1)
                 return false;

             if (a_t1 == b_t1 && a_t2 > b_t2)
                 return false;

             if (a_t1 == b_t1 && a_t2 == b_t2 && a_t3 > b_t3)
                 return false;
             return true;
         }
    );
}

static
void
rand_rgb(int *rgb)
{//随机产生颜色
    rgb[0] = rand() % 255;
    rgb[1] = rand() % 255;
    rgb[2] = rand() % 255;
}

static std::string
get_name(std::string s)
{

    int start_pos = s.find_last_of("/");
    int stop_pos = s.find_last_of(".");
    std::string s_out = s.substr(start_pos + 1, stop_pos - start_pos - 1);
    return s_out;
}

#endif //CHECKERBOARD_LC_CALIB_UTILS_H
