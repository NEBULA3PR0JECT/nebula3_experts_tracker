# nebula3_experts_tracker
Nebula Tracker expert repository
1. Docker - directory with docker files
2. tracker - code for tracker expert 
3. nebula3_database - submodule with database integration and API
# How to create submodule:
1. Get the main module initially  
git clone https://github.com/dsivov/nebula3_experts_tracker.git  
cd nebula3_experts_tracker  
Then get submodule  
git submodule add https://github.com/dsivov/nebula3_database.git    
git submodule init  
2. Change to the submodule directory  
cd nebula3_database 
3. Checkout desired branch  
git checkout master 
4. Update  
git pull 
5. Get back to your project root   
cd ..  
6. Now the submodules are in the state you want, so  
git commit -am "Pulled down update to submodule_dir" 
