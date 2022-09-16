#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <sstream>

using namespace std;

tuple< vector<vector<vector<string>>>, vector<vector<vector<double>>> > xyz_reader(string file, int clinelength = 60);

int main(){
	 auto[atomnames, traj] = xyz_reader("pos.xyz", 60);
}

tuple< vector<vector<vector<string>>>, vector<vector<vector<double>>> > xyz_reader(string file, int clinelength){
	
	// most declarations go here
	double x, y, z; 
	int nlines = 0;
	int nframes = 0;
	int natoms = 0;
	string name, line;

	// reading xyz file as a string and writing temp xyz without comments. 
	ifstream xyz(file); 
 	
	// reading natoms and setting the file pointer location to the start of the file again. 	
	xyz >> natoms;
	xyz.clear();
	xyz.seekg(0);

	// A quick way to split strings separated via any character delimiter using lambda expr.
	auto tokenize = [](string s, char del,string &namei,double &xi,double &yi,double &zi){
		stringstream ss(s);
		string word;
		//double xi, yi, zi = 0.0;
		//string namei;
	  	getline(ss, word, del) >> namei;
	  	getline(ss, word, del) >> xi;
	  	getline(ss, word, del) >> yi;
	  	getline(ss, word, del) >> zi;
	};

	// allocating arrays and counting nframes. 
	//nframes = (nlines/natoms);
	nframes =100;
	natoms =100;
	string X;
	vector<vector<vector<double>>> traj(nframes, vector<vector<double>>(natoms, vector<double>(3, 0)));
	vector<vector<vector<string>>> atomnames(nframes, vector<vector<string>>(natoms, vector<string>(1, X)));

	// saving data from file into vectors. 
	for(int time=0;time<nframes;time++){
		for(int atom=0;atom<natoms;atom++){
			getline(xyz, line);
			if (line.length() > clinelength){
				tokenize(line,' ', name, x, y, z);
				traj[time][atom][0] = x;
				traj[time][atom][1] = y;
				traj[time][atom][2] = z;
				atomnames[time][atom][0] = name;
				// cout << atomnames[time][atom][0] << "  "  << traj[time][atom][0] << "  " << traj[time][atom][1] << "  " << traj[time][atom][2] << endl;
			}	
		}
		//cout << endl;
		//cout << endl;
		//cout << endl;
	}
	
	xyz.close();
	return make_tuple(atomnames, traj);
}

