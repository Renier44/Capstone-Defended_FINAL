import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import dr1 from '../assets/images/dr1.jpg';
import dr2 from '../assets/images/dr2.jpg';
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Alert,
} from 'react-native';

/**
 * Maps the full name of a doctor (as received from the API) to the local image asset.
 * This is the recommended way to handle local asset lookup in React Native.
 */
const DOCTOR_IMAGE_MAP = {
    'Dr. Mikaela Cherry Lopez': dr2,
    'Dr. Maria Cherry Lopez': dr1,
};


export default function ScheduleScreen() {
  const router = useRouter();

  const {
    firstName,
    lastName,
    email,
    dob,
    gender,
    age,
    reason: initialReason,
    bookingFor,
    isEdit,
    id,
    appointmentDate,
    selectedTime: prevTime,
    aiScreening,
  } = useLocalSearchParams();

  const [doctorData, setDoctorData] = useState(null);
  const [availableTimeSlots, setAvailableTimeSlots] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);

  const [currentMonth, setCurrentMonth] = useState(new Date().getMonth());
  const [currentYear, setCurrentDate] = useState(new Date().getFullYear());

  const parsedAge = parseInt(age);

  useEffect(() => {
    if (isEdit && appointmentDate && prevTime) {
      setSelectedDate(appointmentDate);
      setSelectedTime(prevTime);
    }
  }, [isEdit]);

  const fetchAssignedDoctor = async (date = null) => {
    try {
      setIsLoading(true);
      setError(null);

      const url = new URL('https://capstone-defended-final.onrender.com/api/assigned-availability/');
      url.searchParams.append('age', parsedAge);
      if (date) {
        url.searchParams.append('date', date);
      }

      const response = await fetch(url);
      const data = await response.json();

      if (response.ok && data.success) {
        setDoctorData(data.doctor);
        setAvailableDates(data.available_dates);

        if (date) {
          setAvailableTimeSlots(data.time_slots);
          setSelectedDate(data.selected_date);
        } else if (data.selected_date) {
          setSelectedDate(data.selected_date);
          setAvailableTimeSlots(data.time_slots);
        } else {
          setSelectedDate(null);
          setAvailableTimeSlots([]);
        }
      } else {
        setError(data.error || 'Failed to fetch doctor data.');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAssignedDoctor();
  }, []);

  // --- DERIVED STATE FOR IMAGE ---
  const currentDoctorImage = doctorData 
    ? DOCTOR_IMAGE_MAP[doctorData.full_name] || dr1 // Use dr1 as the default local asset if the name isn't mapped
    : null;

  const handleNext = () => {
    if (!selectedDate || !selectedTime || !doctorData) {
      Alert.alert('Incomplete', 'Please select a date and time.');
      return;
    }

    const finalReason = aiScreening ? 'Preliminary Result' : initialReason;

    // Use the derived image asset for routing
    const finalDoctorImage = currentDoctorImage || dr1; 

    router.push({
      pathname: '/confirm-booking',
      params: {
        firstName,
        lastName,
        email,
        age,
        gender,
        reason: finalReason,
        bookingFor,
        doctorId: doctorData.id,
        assignedDoctorName: doctorData.full_name,
        doctorImage: finalDoctorImage, // Now uses the clean, derived asset
        selectedTime,
        appointmentDate: selectedDate,
        ...(isEdit ? { isEdit, id } : {}),
        aiScreening,
      },
    });
  };


  const renderCalendar = () => {
    const firstDayOfMonth = new Date(currentYear, currentMonth, 1).getDay();
    const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    // Adjust start day: Sunday (0) should be last, Monday (1) first
    const adjustedFirstDay = firstDayOfMonth === 0 ? 6 : firstDayOfMonth - 1; 

    const blanks = Array(adjustedFirstDay).fill(null);
    const daysArray = Array.from({ length: daysInMonth }, (_, i) => i + 1);
    const calendarDays = [...blanks, ...daysArray];

    const goToPreviousMonth = () => {
      let newMonth = currentMonth - 1;
      let newYear = currentYear;
      if (newMonth < 0) {
        newMonth = 11;
        newYear -= 1;
      }
      setCurrentMonth(newMonth);
      setCurrentDate(newYear);
      fetchAssignedDoctor(`${newYear}-${String(newMonth + 1).padStart(2, '0')}-01`);
    };

    const goToNextMonth = () => {
      let newMonth = currentMonth + 1;
      let newYear = currentYear;
      if (newMonth > 11) {
        newMonth = 0;
        newYear += 1;
      }
      setCurrentMonth(newMonth);
      setCurrentDate(newYear);
      fetchAssignedDoctor(`${newYear}-${String(newMonth + 1).padStart(2, '0')}-01`);
    };

    const monthNames = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December',
    ];

    return (
      <View style={styles.calendarContainer}>
        <View style={styles.calendarHeader}>
          <TouchableOpacity onPress={goToPreviousMonth}>
            <Ionicons name="chevron-back-outline" size={24} color="#2260FF" />
          </TouchableOpacity>
          <Text style={styles.monthText}>
            {' '}
            {monthNames[currentMonth].toUpperCase()} {currentYear}{' '}
          </Text>
          <TouchableOpacity onPress={goToNextMonth}>
            <Ionicons name="chevron-forward-outline" size={24} color="#2260FF" />
          </TouchableOpacity>
        </View>

        <View style={styles.weekDays}>
          {['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'].map((day) => (
            <Text key={day} style={styles.weekDayText}>
              {day}
            </Text>
          ))}
        </View>

        <View style={styles.dateGrid}>
          {calendarDays.map((day, index) => {
            if (!day) {
              return <View key={`blank-${index}`} style={styles.dateCell} />;
            }

            const dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            const isAvailable = availableDates.includes(dateStr);
            const isSelected = dateStr === selectedDate;
            const today = new Date();
            const cellDate = new Date(currentYear, currentMonth, day);
            const isPastDate = cellDate < today && cellDate.toDateString() !== today.toDateString();

            const isCurrentDate = cellDate.toDateString() === today.toDateString() && currentMonth === today.getMonth() && currentYear === today.getFullYear();
            
            return (
              <TouchableOpacity
                key={day}
                style={[
                  styles.dateCell,
                  // Inline style used to set background/margin for the date cell wrapper
                  { backgroundColor: '#FFFFFF', borderRadius: 15, marginHorizontal: 3, marginVertical: 0 }, 
                  isSelected && styles.selectedDateCell,
                  (isPastDate || !isAvailable) && styles.disabledDateCell,
                ]}
                onPress={() => {
                  if (isAvailable && !isPastDate) {
                    setSelectedTime(null);
                    fetchAssignedDoctor(dateStr);
                  }
                }}
                disabled={!isAvailable || isPastDate}
              >
                <Text
                  style={[
                    styles.dateText,
                    // If available and not selected/today, use the darker color
                    isAvailable && !isSelected && !isCurrentDate && { color: '#555555' }, 
                    isSelected && styles.selectedDateText,
                    (isPastDate || !isAvailable) && styles.disabledDateText,
                    isCurrentDate && { color: '#2260FF', fontWeight: 'bold', fontSize: 18 }, 
                  ]}
                >
                  {day}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>
    );
  };

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2260FF" />
        <Text style={{ marginTop: 10, fontFamily: 'LeagueSpartan' }}>Finding the right doctor...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={{ color: 'red', textAlign: 'center', fontFamily: 'LeagueSpartan' }}>{error}</Text>
      </View>
    );
  }

  // --- JSX structure updated to match the image layout ---
  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 30 }}>
      {/* Header (Back and Schedule Title) */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#000" />
        </TouchableOpacity>
        <View style={styles.scheduleTitleContainer}>
          <Ionicons name="calendar-outline" size={18} color="#2260FF" />
          <Text style={styles.scheduleTitleText}>Schedule</Text>
        </View>
      </View>

      {/* Doctor Card Section */}
      <View style={styles.doctorCard}>
        <View style={styles.doctorInfoRow}>
          {/* Profile Image (Using the derived variable for clean access) */}
          <Image
            source={
                currentDoctorImage 
                ? currentDoctorImage // Use the local asset if mapped
                : {
                    // Fallback to UI-Avatars if no local image is mapped for this doctor
                    uri: `https://ui-avatars.com/api/?name=${doctorData.full_name}&background=2260FF&color=fff&size=120&bold=true`,
                  }
            }
            style={styles.profileImage}
          />

          {/* Experience and Focus Box */}
          <View style={styles.focusContainer}>
            <View style={styles.experienceBubble}>
              <Ionicons name="bulb-outline" size={16} color="#000" />
              <Text style={styles.experienceText}>10 years experience</Text>
            </View>
            <Text style={styles.focusText}>
              Focus: specializes in vision therapy and non-surgical management of strabismus in children and adolescents (18 years and below).
            </Text>
          </View>
        </View>

        {/* Doctor Name and Title */}
        <Text style={styles.doctorName}>
          {doctorData.full_name}
        </Text>
        <Text style={styles.doctorTitle}>
          {doctorData.specialty}
        </Text>

        {/* Working Hours/Availability */}
        <View style={styles.availabilityChip}>
          <Ionicons name="time-outline" size={16} color="#2260FF" />
          <Text style={styles.availabilityText}>Mon - Sat / 9 AM - 4 PM</Text>
        </View>
      </View>

      {/* Calendar Section */}
      <View style={styles.calendarSection}>
        {renderCalendar()}
      </View>

      {/* Select Time Section */}
      <Text style={styles.selectTimeLabel}>Select Time</Text>
      <View style={styles.timeSlotGrid}>
        {availableTimeSlots.length > 0 ? (
          availableTimeSlots.map((item) => (
            <TouchableOpacity
              key={item}
              style={[styles.timeSlot, selectedTime === item && styles.selectedTimeSlot]}
              onPress={() => setSelectedTime(item)}
            >
              <Text style={[styles.timeText, selectedTime === item && styles.selectedTimeText]}>
                {item}
              </Text>
            </TouchableOpacity>
          ))
        ) : (
          <Text style={styles.noSlotsText}>No slots available on this date.</Text>
        )}
      </View>

      {/* Next Button */}
      <TouchableOpacity
        style={[
          styles.nextButton,
          (!selectedTime || !selectedDate) && styles.disabledNextButton,
        ]}
        onPress={handleNext}
        disabled={!selectedTime || !selectedDate}
      >
        <Text style={styles.nextButtonText}>NEXT</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#C8EAF7', // Light blue background matching the image
    paddingHorizontal: 20,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  // --- Header Styles ---
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 50, // To push content down from the top edge
    marginBottom: 20,
  },
  backButton: {
    paddingRight: 15,
  },
  scheduleTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E0F7FA', // Light blue background for the chip
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#99E0E9', // Border color
  },
  scheduleTitleText: {
    fontSize: 14,
    color: '#2260FF',
    fontWeight: '600',
    marginLeft: 5,
    fontFamily: 'LeagueSpartan',
  },
  // --- Doctor Card Styles ---
  doctorCard: {
    backgroundColor: '#fff', // White background
    borderRadius: 25,
    padding: 20,
    alignItems: 'center',
    marginBottom: 20,
    // Shadow to match the elevated card look
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 8,
  },
  doctorInfoRow: {
    flexDirection: 'row',
    marginBottom: 10,
    alignItems: 'flex-start',
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
    marginRight: 15,
  },
  focusContainer: {
    flex: 1,
    backgroundColor: '#FFEB3B', // Bright yellow background
    borderRadius: 15,
    padding: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  experienceBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
    backgroundColor: '#FFF9C4', // Lighter yellow bubble inside focus box
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: 10,
    alignSelf: 'flex-start',
  },
  experienceText: {
    fontSize: 12,
    fontWeight: 'bold',
    marginLeft: 4,
    fontFamily: 'LeagueSpartan',
    color: '#000',
  },
  focusText: {
    fontSize: 12,
    color: '#333',
    fontFamily: 'LeagueSpartan',
  },
  doctorName: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2260FF',
    fontFamily: 'LeagueSpartan',
    marginTop: 5,
  },
  doctorTitle: {
    fontSize: 16,
    color: '#555',
    marginBottom: 10,
    fontFamily: 'LeagueSpartan',
  },
  availabilityChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E0F7FA', // Light blue background for the chip
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#99E0E9',
  },
  availabilityText: {
    fontSize: 14,
    color: '#2260FF',
    fontWeight: '600',
    marginLeft: 5,
    fontFamily: 'LeagueSpartan',
  },
  // --- Calendar Styles ---
  // --- Calendar Styles ---
  calendarSection: {
    marginBottom: 20,
    borderRadius: 25,
    overflow: 'hidden', 
    backgroundColor: '#D1F0F7', 
  },
  calendarContainer: {
    width: '100%',
    backgroundColor: '#77CDE0', 
    borderRadius: 25,
    paddingHorizontal: 5, 
    paddingVertical: 2, // REDUCED from 5 to 2
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 0,
  },
  calendarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2, // REDUCED from 5 to 2
    paddingHorizontal: 5, 
    paddingVertical: 2, // REDUCED from 5 to 2
  },
  monthText: {
    fontWeight: 'bold',
    fontSize: 18,
    color: '#2260FF', 
    fontFamily: 'LeagueSpartan',
    marginTop: 0, 
  },
  weekDays: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 2, // REDUCED from 5 to 2
  },
  weekDayText: {
    color: '#FFFFFF', 
    backgroundColor: '#2260FF', 
    paddingVertical: 2, // REDUCED from 4 to 2
    paddingHorizontal: 4,
    borderRadius: 15,
    fontSize: 12,
    textAlign: 'center',
    width: '12%',
    fontWeight: 'bold',
    fontFamily: 'LeagueSpartan',
  },
dateGrid: {
  flexDirection: 'row',
  flexWrap: 'wrap',
  justifyContent: 'space-between',
  backgroundColor: '#FFFFFF',
  borderRadius: 20,
  padding: 10,
},
// ... rest of the styles
dateCell: {
  paddingHorizontal: 20,     // fits the date closely
  paddingVertical: 5,
  borderRadius: 8,
  backgroundColor: '#FFFFFF',
  justifyContent: 'center',
  alignItems: 'center',
  marginVertical: 3,
},
  dateText: {
    color: '#A3A3A3',
    fontSize: 14,
    fontFamily: 'LeagueSpartan',
  },
  selectedDateCell: {
    backgroundColor: '#2260FF', // Dark blue background for selected date
  },
  selectedDateText: {
    color: '#FFFFFF', // White text for selected date
    fontWeight: 'bold',
  },
  disabledDateCell: {
    // No explicit background for disabled, relies on text color for visual cue
  },
  disabledDateText: {
    color: '#A9A9A9', // Light gray text for unavailable/past dates
  },
  // --- Time Slot Styles ---
  selectTimeLabel: {
    fontWeight: 'bold',
    marginBottom: 15,
    fontSize: 16,
    color: '#2260FF',
    fontFamily: 'LeagueSpartan',
    textAlign: 'center',
  },
  timeSlotGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 10,
    width: '100%',
    marginBottom: 30,
  },
  timeSlot: {
    backgroundColor: '#E6E6FA', // Light purple/lavender background for available slots
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20, // Pill shape
    borderWidth: 1,
    borderColor: '#D0E0FF', // Light border
  },
  selectedTimeSlot: {
    backgroundColor: '#2260FF', // Dark blue background for selected slot
    borderColor: '#2260FF',
  },
  timeText: {
    color: '#2260FF', // Dark blue text for available slots
    fontSize: 13,
    fontWeight: '500',
    fontFamily: 'LeagueSpartan',
  },
  selectedTimeText: {
    color: '#FFFFFF', // Changed to White text for selected time to match the provided image
    fontWeight: 'bold',
  },
  noSlotsText: {
    textAlign: 'center',
    color: '#666',
    fontFamily: 'LeagueSpartan',
  },
  // --- Next Button Styles ---
  nextButton: {
    backgroundColor: '#FFD54F', // Approximation of the light blue gradient from the image
    paddingVertical: 13,
    borderRadius: 25,
    alignItems: 'center',
    marginBottom: 30,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 6,
    elevation: 8,
  },
  disabledNextButton: {
    backgroundColor: '#A9A9A9', // Grey background for disabled state
  },
  nextButtonText: {
    
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 18,
    fontFamily: 'LeagueSpartan',
  },
});