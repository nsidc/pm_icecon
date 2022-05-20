#!/usr/bin/ksh
#!/home/ndigirol/bin/ksh
# not available on oibserve: !/bin/ksh

#====================================================================
#    ***** edit the script  run_icecons.ksh *****
#
#    1 set the years to year of ice concentration data to be produced
#
#    2 check TB_HOME directory setting (/pmw if using archive location)
#
#    3 set the channels to those for the intrument (eg, f13 vs f17)
#
#    4 set day1 and day2 to start/end day of year being processed
#
#    5 set "ssss1dXXtb", where dXX is d17 or d13, etc (search for ssss1d below)
#
#    6 set "TOT_CON ssmifXX" where ssmifXX to be ssmif17 or ssmif13, etc (search for TOT_CON below)
#====================================================================

# script to run the icecon system
# Alvaro Ivanoff, Dec. 2002

#*****    . set the years to year of ice concentration data to be produced *****
#set -A years 2008 2009 2010 2011 2012 2013
set -A years 2018

# HOME=/data/users/ndigirol/icecon_code/production/FOR_WALT  # oibserve as of 4/17/18
HOME=/home/scotts/sic_py/nt_orig  # oibserve as of 4/17/18

#*****    . check TB_HOME directory setting (/pmw for archive location) *****
# TB_HOME=/data/users/ndigirol/icecon_code/production/FOR_WALT/system/0_tbs/  # oibserve as of 4/17/18
TB_HOME=/home/scotts/sic_py/nt_orig/system/0_tbs/  # oibserve as of 4/17/18
#*****   end check TB_HOME directory setting (/pmw for archive location) *****

#*****    . set the channels to those for the intrument  *****
set -A poles n s

# f13 and prior use this
#set -A channels 19h 19v 22v 37h 37v 85h 85v

# f17 and later use this
set -A channels 19h 19v 22v 37h 37v 91h 91v
#*****  end set the channels to those for the intrument *****

SPATIALINT_DIR=$HOME/system/1_tbs_spatially_filtered


echo "TB Input Directory (TB_HOME): $TB_HOME"
echo "Years to process: " ${years[*]} 
echo "Channels: " ${channels[*]}
echo " "

cd $HOME

for year in ${years[*]}
do

#*****    . 
# set day1 and day2 to start/end day of year being processed *****
  
  ## the range of days that gets processed is day1 to day2
  ## Should always start with day1 set to 001 normally.
  day1=001

  ## day2 depends on whether the current year is a leap year.
  ## this sets day2 to the number of days in the year
  #day2=`echo $year 'sY 3 lY 4%- 3/ 273+p' | dc` 
  #day2=`echo $year 'sY 3 lY 4%- 3/ 365+p' | dc` 

  # 08/31/2015 - running for all available 2015 days 1-181
  #day2=181
  # 03/14/2016 - running for all available 2015 days 1-365

  day2=010
  #365
  
#*****   end set day1 and day2 to start/end day of year being processed *****

 echo "Year: " $year
 echo "Days " $day1 $day2
 
 # read ans?"Continue? "

 # echo $ans
 # if [ $ans == n ] 
 # then 
 #  echo "no"
 #  echo "Exiting"
 #  exit
 # fi
 #   
 # if [ $ans == y ] 
 # then 
 #  echo "yes"
 #  echo "Running..."
 # fi
 
    echo "Skipping spatial interpolation"
    # Skipping spatial interpolation
  # for pole in ${poles[*]}
  # do
  #   case $pole in
  #   "n")TB_DIR=$TB_HOME/NSSSS1DTB01;;
  #   "s")TB_DIR=$TB_HOME/SSSSS1DTB01;;
  #   esac    
    # for channel in ${channels[*]}
    # do    
    #   ## create the list of files which goes into the spatial interpolation program
    #   ## the first line must contain the number of files
    #   
    #   filelist=$HOME/system/file_lists/filelist_${pole}_${channel}_${year}.txt      
    #   count=0
    #   for file in $TB_DIR/${pole}ssss1d17tb${channel}${year}*
    #   do
    #     (( count=count+1 ))
    #   done
    #   print $count > $filelist
    #   #print 5 > $filelist

    #   for file in $TB_DIR/${pole}ssss1d17tb${channel}${year}*
    #   do
    #     print $file >> $filelist
    #   done

    #   
    #   ## run the spatial interpolation program

    #   if [[ $channel == 85h||$channel == 85v||$channel == 91h||$channel == 91v ]]
    #   then
    #     $HOME/bin/SpatialInt_${pole}p85<<args
    #     $channel
    #     $filelist
    #     $SPATIALINT_DIR
args# 
    #   else
    #    $HOME/bin/SpatialInt_${pole}p <<args
    #     $channel
    #     $filelist
    #     $SPATIALINT_DIR
args# 
    #   fi
    # done
  # done
  
#exit # test
 
  ## go to the directory where the icecons will be produced
  cd $HOME/system/2_iceconcentrations
  
  echo "$HOME/bin/seaice5con"
  
  $HOME/bin/seaice5con $day1 $year $day2 $year TOT_CON ssmif17 n
  
  $HOME/bin/seaice5con $day1 $year $day2 $year TOT_CON ssmif17 s

  find . -name "n*tcon${year}*" | sort |  ls -1

  # find . -name "n*tcon${year}*" | sort |  $HOME/bin/apply_sst_n
  pwd
  ls n*tcon${year}* | sort |  $HOME/bin/apply_sst_n

  # find . -name "s*${year}*" | sort |  $HOME/bin/apply_sst_s
  ls s*tcon${year}* | sort |  $HOME/bin/apply_sst_s
  
  find . -name "*.spill_sst"|xargs -i mv {} $HOME/system/3_icecons_spill_sst

done
