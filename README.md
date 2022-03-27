# nebula3_experts_tracker
Nebula Tracker expert repository
1. Docker - directory with docker files
2. tracker - code for tracker expert 
3. nebula3_database - submodule with database integration and API
# How to create submodule:
Get the submodule initially
git clone https://github.com/dsivov/nebula3_experts_tracker.git
cd nebula3_experts_tracker
git submodule add https://github.com/dsivov/nebula3_database.git
git submodule init
Change to the submodule directory
cd nebula3_database
Checkout desired branch
git checkout master
Update
git pull
Get back to your project root
cd ..
Now the submodules are in the state you want, so
git commit -am "Pulled down update to submodule_dir"
