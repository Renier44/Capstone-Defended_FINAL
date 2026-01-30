import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
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

export default function ScheduleScreen() {
  const router = useRouter();
  const {
    firstName,
    lastName,
    dob,
    gender,
    age,
    reason,
    bookingFor,
  } = useLocalSearchParams();

  const [doctorData, setDoctorData] = useState(null);
  const [availableTimeSlots, setAvailableTimeSlots] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);

  // Dynamic month/year state
  const [currentMonth, setCurrentMonth] = useState(new Date().getMonth());
  const [currentYear, setCurrentYear] = useState(new Date().getFullYear());

  const parsedAge = parseInt(age);

  const fetchAssignedDoctor = async (date = null) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const url = new URL('https://2b7bf55b1e09.ngrok-free.app/api/assigned-availability/');
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

  const handleNext = () => {
    if (!selectedDate || !selectedTime || !doctorData) {
      Alert.alert('Incomplete', 'Please select a date and time.');
      return;
    }

    router.push({
      pathname: '/confirm-booking',
      params: {
        firstName,
        lastName,
        age,
        gender,
        reason,
        bookingFor,
        doctorId: doctorData.id,
        assignedDoctorName: doctorData.full_name,
        selectedTime,
        appointmentDate: selectedDate,
      },
    });
  };

  const renderCalendar = () => {
    const daysInCurrentMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    const calendarDays = Array.from({ length: daysInCurrentMonth }, (_, i) => i + 1);

    const goToPreviousMonth = () => {
      let newMonth = currentMonth - 1;
      let newYear = currentYear;
      if (newMonth < 0) {
        newMonth = 11;
        newYear -= 1;
      }
      setCurrentMonth(newMonth);
      setCurrentYear(newYear);
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
      setCurrentYear(newYear);
      fetchAssignedDoctor(`${newYear}-${String(newMonth + 1).padStart(2, '0')}-01`);
    };

    return (
      <View style={styles.calendarContainer}>
        <View style={styles.calendarHeader}>
          <TouchableOpacity onPress={goToPreviousMonth}><Text style={styles.arrow}>{'<'}</Text></TouchableOpacity>
          <Text style={styles.monthText}>
            {new Date(currentYear, currentMonth).toLocaleString('default', { month: 'long', year: 'numeric' })}
          </Text>
          <TouchableOpacity onPress={goToNextMonth}><Text style={styles.arrow}>{'>'}</Text></TouchableOpacity>
        </View>

        <View style={styles.weekDays}>
          {['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'].map(day => (
            <Text key={day} style={styles.weekDay}>{day}</Text>
          ))}
        </View>

        <View style={styles.dateGrid}>
          {calendarDays.map(day => {
            const dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            const isAvailable = availableDates.includes(dateStr);
            const isSelected = dateStr === selectedDate;

            return (
              <TouchableOpacity
                key={day}
                style={[
                  styles.dateCell,
                  isSelected && styles.selectedDateCell,
                  !isAvailable && styles.disabledDateCell,
                ]}
                onPress={() => {
                  if (isAvailable) {
                    setSelectedTime(null);
                    fetchAssignedDoctor(dateStr);
                  }
                }}
                disabled={!isAvailable}
              >
                <Text style={[
                  styles.dateText,
                  isSelected && styles.selectedDateText,
                  !isAvailable && styles.disabledDateText,
                ]}>{day}</Text>
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
        <ActivityIndicator size="large" color="#1E88E5" />
        <Text style={{ marginTop: 10 }}>Finding the right doctor...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={{ color: 'red', textAlign: 'center' }}>{error}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 30 }}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#2260FF" />
        </TouchableOpacity>
        <Text style={styles.heading}>Select Schedule</Text>
      </View>

      <View style={styles.card}>
        <Image source={{ uri: `https://ui-avatars.com/api/?name=${doctorData.full_name}&background=2260FF&color=fff&size=120&bold=true` }} style={styles.profileImage} />
        <Text style={styles.doctorName}>{doctorData.full_name}</Text>
        <Text style={styles.experience}>Specialty: {doctorData.specialty}</Text>
        <Text style={styles.availability}>Availability: based on calendar</Text>

        {renderCalendar()}

        <Text style={styles.selectTimeLabel}>Select Time:</Text>
        <View style={styles.timeSlotGrid}>
          {availableTimeSlots.length > 0 ? (
            availableTimeSlots.map((item) => (
              <TouchableOpacity
                key={item}
                style={[styles.timeSlot, selectedTime === item && styles.selectedTimeSlot]}
                onPress={() => setSelectedTime(item)}
              >
                <Text style={[styles.timeText, selectedTime === item && styles.selectedTimeText]}>{item}</Text>
              </TouchableOpacity>
            ))
          ) : (
            <Text style={{ textAlign: 'center', color: '#666' }}>No slots available on this date.</Text>
          )}
        </View>
      </View>

      <TouchableOpacity
        style={[styles.nextButton, (!selectedTime || !selectedDate) && { backgroundColor: '#ccc' }]}
        onPress={handleNext}
        disabled={!selectedTime || !selectedDate}
      >
        <Text style={styles.nextButtonText}>Next</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  // Note: Custom fonts in React Native require additional steps (like adding the font file to the project)
  // which is not possible in this single-file environment. The 'LeagueSpartan' font family is added
  // here to demonstrate where the change would be applied if the font were available.
  container: { flex: 1, padding: 20, backgroundColor: '#77CDE0' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  header: { flexDirection: 'row', alignItems: 'center', marginBottom: 60 },
  heading: { fontSize: 24, fontWeight: 'bold', color: '#2260FF', textAlign: 'center', flex: 1, fontFamily: 'LeagueSpartan' },
  card: { backgroundColor: '#C8EAF7', borderRadius: 15, padding: 20, alignItems: 'center', marginBottom: 40, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 3 },
  profileImage: { width: 120, height: 120, borderRadius: 60, marginBottom: 10 },
  doctorName: { fontSize: 20, fontWeight: 'bold', color: '#2260FF', fontFamily: 'LeagueSpartan' },
  experience: { color: '#f9a825', marginVertical: 5, fontWeight: '500', fontFamily: 'LeagueSpartan' },
  availability: { color: '#555', fontSize: 14, marginBottom: 10, fontFamily: 'LeagueSpartan' },
  calendarContainer: { width: '100%', backgroundColor: '#77CDE0', borderRadius: 15, padding: 10, marginBottom: 15, borderWidth: 1, borderColor: '#FFD54F' },
  calendarHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  monthText: { fontWeight: 'bold', fontSize: 16, color: '#2260FF', fontFamily: 'LeagueSpartan' },
  arrow: { fontSize: 18, color: '#2260FF', fontFamily: 'LeagueSpartan' },
  weekDays: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  weekDay: { color: '#fff', backgroundColor: '#2260FF', paddingVertical: 4, paddingHorizontal: 6, borderRadius: 10, fontSize: 10, textAlign: 'center', fontFamily: 'LeagueSpartan' },
  dateGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'flex-start' },
  dateCell: { width: 34, height: 34, margin: 4, borderRadius: 17, backgroundColor: '#FFFFFF', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: '#FFFFFF' },
  selectedDateCell: { backgroundColor: '#2260FF', borderColor: '#2260FF' },
  dateText: { color: '#000000', fontSize: 12, fontFamily: 'LeagueSpartan' },
  selectedDateText: { color: '#FFD54F', fontWeight: 'bold', fontFamily: 'LeagueSpartan' },
  disabledDateCell: { backgroundColor: '#D3D3D3', borderColor: '#A9A9A9' },
  disabledDateText: { color: '#666', fontFamily: 'LeagueSpartan' },
  selectTimeLabel: { fontWeight: 'bold', marginVertical: 10, fontSize: 16, color: '#2260FF', fontFamily: 'LeagueSpartan' },
  timeSlotGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 8, width: '100%' },
  timeSlot: { backgroundColor: '#FFD54F', padding: 10, marginVertical: 4, borderRadius: 10, alignItems: 'center', borderWidth: 1, borderColor: '#FFD54F' },
  selectedTimeSlot: { backgroundColor: '#2260FF', borderColor: '#2260FF' },
  timeText: { color: '#2260FF', fontSize: 12, fontFamily: 'LeagueSpartan' },
  selectedTimeText: { color: '#FFD54F', fontWeight: 'bold', fontFamily: 'LeagueSpartan' },
  nextButton: { backgroundColor: '#2260FF', paddingVertical: 15, borderRadius: 10, alignItems: 'center', marginTop: 40, marginBottom: 30, shadowColor: '#000', shadowOpacity: 0.15, shadowOffset: { width: 0, height: 2 }, shadowRadius: 4, elevation: 3 },
  nextButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 16, fontFamily: 'LeagueSpartan' },
});
