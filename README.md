# nebula3_experts_tracker
Nebula Tracker expert repository
1. Docker - directory with docker files
2. tracker - code for tracker expert 
3. nebula3_database - submodule with database integration and API
# How to create submodule:
1. Get the main module initially:  
`git clone https://github.com/dsivov/nebula3_experts_tracker.git  `
`cd nebula3_experts_tracker `

2. Then get submodule:  
`git submodule add https://github.com/dsivov/nebula3_database.git`    
`git submodule init `

3. Change to the submodule directory:  
`cd nebula3_database` 

4. Checkout desired branch:  
`git checkout main` 

5. Update:  
git pull 

6. Get back to your project root:   
`cd ..`  

7. Now the submodules are in the state you want, so"  
`git commit -am "Pulled down update to submodule_dir"` 
