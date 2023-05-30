#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

int main()
{
    
    char buf[1024];
    FILE *temp;

    const int max_threads = 1024;
    
    // Initializing CSV
    std::ofstream fp("load_model.csv");
    // Writing CSV Header
    fp << "Warps" << "," << "Conflicts" << "," << "Time" << std::endl;
    for (int i = 32; i <= max_threads; i = i + 32)
    {
        //for (int j = 0; j <= (i % 32); j++)
        for(int j = 0; j <= 31; j++)
        {
            //const int threads = 32;
            //const int conflicts = 31;
            std::stringstream ss;
            std::string line, str;
            char buf[1024];
            str = "microloads.exe ";
            str = str + std::to_string(i) + " " + std::to_string(j) + " > temp.txt";
            const char *command = str.c_str();
            //std::cout << "Command is " << command << endl;
            system(command);
            temp = std::fopen("temp.txt", "r");
            while(fgets(buf, 1024, temp))
            {
                //std::cout << buf << std::endl;
                ss << buf;
            }
            while (std::getline(ss, line)) {
                //std::cout << line << std::endl;
                fp << line << endl;
            }
            fclose(temp);
        }
    }
    fp.close();
    return 0;

}