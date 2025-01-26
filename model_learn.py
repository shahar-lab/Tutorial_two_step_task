# -*- coding: utf-8 -*-

# Copyright (C) 2019  Carolina Feher da Silva

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Runs the two-stage task with the magic carpet story."""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import argparse
import sys
import os
import socket
import random
import csv
import io
import time
from itertools import chain
from os.path import join
from psychopy import visual, core, event, data, gui
import wx
from bidi.algorithm import get_display  # For proper RTL text handling

# Directories
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = join(CURRENT_DIR, 'assets')
RESULTS_DIR = join(CURRENT_DIR, 'tutorial_results')

# CHANGE PARAMETER BELOW BEFORE RUNNING
# Font for displaying the instructions
TTF_FONT = join(CURRENT_DIR, 'OpenSans-SemiBold.ttf')
# Mapping of English colors to Hebrew
color_translations = {
    'red': u'האדום',
    'black': u'השחור',
    'pink': u'הורוד',
    'blue': u'הכחול'
}
# Configuration for tutorial and game
class TutorialConfig:
    final_state_colors = ('red', 'black')
    initial_state_symbols = (7, 8)
    final_state_symbols = ((9, 10), (11, 12))
    num_trials = 20 #20
    common_prob = 0.7 # Optimal performance 61%
    @classmethod
    def proceed(cls, trials, slow_trials):
        del slow_trials
        return trials < cls.num_trials
    @classmethod
    def do_break(cls, trials, slow_trials):
        del trials, slow_trials
        return False
    @classmethod
    def get_common(cls, trial):
        if trial == 0 or trial == 1:
            return True
        if trial == 2:
            return False
        return random.random() < cls.common_prob

# Classes and functions

def main():

    # Get participant information
    info = {
    'subject_number': ''
    }
    dlg = gui.DlgFromDict(info, title='Participant information')
    if not dlg.OK:
        core.quit()
    part_code = info['subject_number'].strip().upper()
    if not part_code:
        print('Empty subject_number')
        core.quit()

    #create folder for results
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    #create filename
    filename = join(
        RESULTS_DIR, '{}_{}'.format(part_code, data.getDateStr()))
    if part_code == 'TEST':
        fullscr = False # Displays small window
        # Decrease number of trials
        TutorialConfig.num_trials = 20
    else:
        fullscr = True # Fullscreen for the participants

    # Create window
    win = visual.Window(
        fullscr=fullscr, size=[800,600], units='pix', color='#404040',gamma=None)
    win.mouseVisible = False

    # Load all images
    images = load_image_collection(win, ASSETS_DIR)

    # Randomize mountain sides and common transitions for the tutorial and game
    tutorial_mountain_sides = list(TutorialConfig.final_state_colors)
    random.shuffle(tutorial_mountain_sides)
    tutorial_model = Model.create_random(TutorialConfig)
    
    # Tutorial flights
    with io.open('{}_tutorial.csv'.format(filename), 'wb') as outf:
        csv_writer = csv.DictWriter(outf, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        run_trial_sequence(
            TutorialConfig, TutorialDisplay(win, images, tutorial_mountain_sides),
            tutorial_model, csv_writer)

    
    # Display Hebrew text
    finish_text = visual.TextStim(win, text=u" הניסוי הסתיים, תודה!"[::-1], font='Arial')
    finish_text.draw()
    win.flip()

    # Wait for spacebar
    print("Waiting for spacebar press...")
    event.waitKeys(keyList=('space',))
    print("Spacebar pressed!")

    # Close the window
    win.close()
    core.quit()  # Quit PsychoPy


class RewardProbability(float):
    "Reward probability that drifts within a min and a max value."
    MIN_VALUE = 0.25
    MAX_VALUE = 0.75
    DIFFUSION_RATE = 0.025
    def __new__(cls, value):
        assert value >= cls.MIN_VALUE and value <= cls.MAX_VALUE
        return super(RewardProbability, cls).__new__(cls, value)
    @classmethod
    def create_random(cls):
        "Create a random reward probability within the allowed interval."
        return cls(random.uniform(cls.MIN_VALUE, cls.MAX_VALUE))
    def diffuse(self):
        "Get the next probability by diffusion."
        return self.__class__(
            self.reflect_on_boundaries(random.gauss(0, self.DIFFUSION_RATE)))
    def get_reward(self):
        "Get a reward (0 or 1) with this probability."
        return int(random.random() < self)
    def reflect_on_boundaries(self, incr):
        "Reflect reward probability on boundaries."
        next_value = self + (incr % 1)
        if next_value > self.MAX_VALUE:
            next_value = 2*self.MAX_VALUE - next_value
        if next_value < self.MIN_VALUE:
            next_value = 2*self.MIN_VALUE - next_value
        return next_value

class Symbol(object):
    "A Tibetan symbol for a carpet or lamp."
    def __init__(self, code):
        self.code = code
    def __str__(self):
        return '{:02d}'.format(self.code)

class InitialSymbol(Symbol):
    "An initial state symbol."
    def __init__(self, code, final_state):
        super(InitialSymbol, self).__init__(code)
        self.final_state = final_state

class FinalSymbol(Symbol):
    "A final state symbol."
    def __init__(self, code, reward_probability):
        super(FinalSymbol, self).__init__(code)
        self.reward_probability = reward_probability
        self.reward = self.reward_probability.get_reward()

class State(object):
    "A initial state in the task."
    def __init__(self, symbols):
        assert len(symbols) == 2
        self.symbols = symbols

class FinalState(State):
    "A final state in the task."
    def __init__(self, color, symbols):
        self.color = color
        super(FinalState, self).__init__(symbols)

class Model(object):
    """A transition model and configuration of final states for the task."""
    def __init__(self, isymbol_codes, colors, fsymbol_codes):
        self.isymbol_codes = isymbol_codes
        self.colors = colors
        self.fsymbol_codes = fsymbol_codes
    @classmethod
    def create_random(cls, config):
        """Create a random model for the task from a given configuration."""
        colors = list(config.final_state_colors)
        random.shuffle(colors)
        fsymbol_codes = list(config.final_state_symbols)
        random.shuffle(fsymbol_codes)
        return cls(config.initial_state_symbols, colors, fsymbol_codes)
    def get_paths(self, common):
        "Generator for the paths from initial symbol to final symbols."
        if common:
            for isymbol_code, color, fsymbol_codes in zip(
                    self.isymbol_codes, self.colors, self.fsymbol_codes):
                yield (isymbol_code, color, fsymbol_codes)
        else:
            for isymbol_code, color, fsymbol_codes in zip(
                    self.isymbol_codes, reversed(self.colors), reversed(self.fsymbol_codes)):
                yield (isymbol_code, color, fsymbol_codes)
    def __str__(self):
        output = "Common transitions: "
        for isymbol_code, color, fsymbol_codes in self.get_paths(True):
            output += "{} -> {} -> {}; ".format(isymbol_code, color, fsymbol_codes)
        return output

class Trial(object):
    "A trial in the task."
    def __init__(self, number, initial_state, common):
        self.number = number
        self.initial_state = initial_state
        self.common = common
    @classmethod
    def get_sequence(cls, config, model):
        "Get an infinite sequence of trials with this configuration."
        trials = 0
        reward_probabilities = {
            fsymbol_code: RewardProbability.create_random()
            for fsymbol_code in chain(*config.final_state_symbols)
        }
        while True:
            common = config.get_common(trials)
            isymbols = []
            for isymbol_code, color, fsymbol_codes in model.get_paths(common):
                fsymbols = [
                    FinalSymbol(fsymbol_code, reward_probabilities[fsymbol_code])
                    for fsymbol_code in fsymbol_codes
                ]
                random.shuffle(fsymbols)
                final_state = FinalState(color, tuple(fsymbols))
                isymbols.append(InitialSymbol(isymbol_code, final_state))
            random.shuffle(isymbols)
            initial_state = State(isymbols)
            yield cls(trials, initial_state, common)
            for fsymbol_code, prob in reward_probabilities.items():
                reward_probabilities[fsymbol_code] = prob.diffuse()
            trials += 1

CSV_FIELDNAMES = (
    'trial', 'common', 'reward.1.1', 'reward.1.2', 'reward.2.1',
    'reward.2.2', 'isymbol_lft', 'isymbol_rgt', 'rt1', 'choice1', 'final_state',
    'fsymbol_lft', 'fsymbol_rgt', 'rt2', 'choice2', 'reward', 'slow')

def get_intertrial_interval():
    #return random.uniform(0.7, 1.3)
    return 1

def code_to_bin(code, common=True):
    if common:
        return 2 - code % 2
    else:
        return code % 2 + 1

def run_trial_sequence(config, display, model, csv_writer):
    rewards = 0
    slow_trials = 0
    common_transitions = {
        isymbol_code: {'color': color}
        for isymbol_code, color, fsymbol_codes in model.get_paths(True)
    }
    # Trial loop
    for trial in Trial.get_sequence(config, model):
        completed_trials = trial.number - slow_trials
        row = {'trial': trial.number, 'common': int(trial.common)}
        for isymbol in trial.initial_state.symbols:
            for fsymbol in isymbol.final_state.symbols:
                key = 'reward.{}.{}'.format(
                    code_to_bin(isymbol.code, trial.common), code_to_bin(fsymbol.code))
                row[key] = fsymbol.reward_probability
        row['isymbol_lft'] = code_to_bin(trial.initial_state.symbols[0].code)
        row['isymbol_rgt'] = code_to_bin(trial.initial_state.symbols[1].code)

        display.display_start_of_trial(trial.number)

        # First-stage choice
        isymbols = [symbol.code for symbol in trial.initial_state.symbols]
        display.display_carpets(completed_trials, isymbols, common_transitions)

        event.clearEvents()
        keys_times = event.waitKeys(
            maxWait=8, keyList=('s', 'k'), timeStamped=core.Clock())

        if keys_times is None:
            slow_trials += 1
            display.display_slow1()
            row.update({
                'rt1': -1,
                'choice1': -1,
                'final_state': -1,
                'fsymbol_lft': -1,
                'fsymbol_rgt': -1,
                'rt2': -1,
                'choice2': -1,
                'reward': 0,
                'slow': 1,
            })
        else:
            choice1, rt1 = keys_times[0]
            row['rt1'] = rt1

            display.display_selected_carpet(completed_trials, choice1, isymbols, common_transitions)

            # Transition
            chosen_symbol1 = trial.initial_state.symbols[int(choice1 == 's')]
            final_state = chosen_symbol1.final_state
            row['choice1'] = code_to_bin(chosen_symbol1.code)
            row['final_state'] = code_to_bin(chosen_symbol1.code, trial.common)

            display.display_transition(completed_trials, final_state.color, trial.common)

            # Second-stage choice
            fsymbols = [symbol.code for symbol in final_state.symbols]
            row['fsymbol_lft'] = code_to_bin(final_state.symbols[0].code)
            row['fsymbol_rgt'] = code_to_bin(final_state.symbols[1].code)

            display.display_lamps(completed_trials, final_state.color, fsymbols)

            event.clearEvents()
            keys_times = event.waitKeys(
                maxWait=8, keyList=('s', 'k'), timeStamped=core.Clock())
            if keys_times is None:
                slow_trials += 1
                display.display_slow2(final_state.color)
                row.update({
                    'rt2': -1,
                    'choice2': -1,
                    'reward': 0,
                    'slow': 1,
                })
            else:
                choice2, rt2 = keys_times[0]
                row['rt2'] = rt2

                display.display_selected_lamp(completed_trials, final_state.color, fsymbols, choice2)

                # Reward
                chosen_symbol2 = final_state.symbols[int(choice2 == 's')]
                row['choice2'] = code_to_bin(chosen_symbol2.code)
                reward = chosen_symbol2.reward
                row['reward'] = reward
                row['slow'] = 0
                if reward:
                    rewards += 1
                    display.display_reward(completed_trials, final_state.color, chosen_symbol2.code)
                else:
                    display.display_no_reward(completed_trials, final_state.color, chosen_symbol2.code)

        display.display_end_of_trial()

        # Break
        if config.do_break(trial.number + 1, slow_trials):
            display.display_break()
            event.waitKeys(keyList=('space',))
        assert all([fdn in row.keys() for fdn in CSV_FIELDNAMES])
        assert all([key in CSV_FIELDNAMES for key in row.keys()])
        csv_writer.writerow(row)
        # Should we run another trial?
        if not config.proceed(trial.number + 1, slow_trials):
            break
    return rewards

def load_image_collection(win, images_directory):
    image_collection = {
        os.path.splitext(fn)[0]: visual.ImageStim(
            win=win,
            pos=(0, 0),
            image=join(images_directory, fn),
            name=os.path.splitext(fn)[0]
        )
        for fn in os.listdir(images_directory) if os.path.splitext(fn)[1] == '.png'
    }
    return image_collection

def get_random_transition_model(config):
    isymbols = list(config.initial_state_symbols)
    fsymbols = list(config.final_state_symbols)
    colors = list(config.final_state_colors)
    random.shuffle(isymbols)
    random.shuffle(fsymbols)
    random.shuffle(colors)
    return {isymbols[i]: {'color': colors[i], 'symbols': fsymbols[i]} for i in xrange(2)}

class TutorialDisplay(object):
    def __init__(self, win, images, mountain_sides):
        self.win = win
        self.images = images
        self.mountain_sides = mountain_sides
        self.visits_to_mountains = {color: 0 for color in TutorialConfig.final_state_colors}
        # Create frame to display messages
        self.msg_frame = visual.Rect(
            win=self.win,
            pos=(0, 400),
            width=1180,
            height=80,
            fillColor=(1.0, 1.0, 1.0),
            opacity=0.9,
            name='Tutorial message frame',
        )
        self.msg_text = visual.TextStim(
            win=self.win,
            pos=(600, 405),
            height=30,
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(-1, -1, -1),
            wrapWidth=1120,
            alignHoriz='right',
            alignVert='center',
            name='Tutorial message text'
        )
        self.center_text = visual.TextStim(
            win=win,
            pos=(0, 0),
            height=80,
            wrapWidth=980,
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(1, 1, 1),
            name='Center text'
        )
    def display_start_of_trial(self, trial):
        hebrew_text =  str(trial + 1)+ u'נסיעת הכנה מספר '[::-1]
        self.center_text.text = hebrew_text

        self.center_text.draw()
        self.win.flip()
        core.wait(3)
    def display_carpets(self, trial, isymbols, common_transitions):
        isymbols_image = self.images['tibetan.{:02d}{:02d}'.format(*isymbols)]
        destination_image = self.images['carpets_to_{}_{}'.format(
            *[common_transitions[symbol]['color'] for symbol in isymbols]
        )]

        def draw_main_images():
            self.images['carpets_tutorial'].draw()
            isymbols_image.draw()
            destination_image.draw()

        if trial < 3:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = hebrew_text = u'הוצאתם את השטיחים המכושפים שלכם מהארון '\
              u'ופרסתם אותם על הרצפה.'[::-1]  # היפוך הכיוון
            self.msg_text.draw()
            self.win.flip()
            core.wait(4.5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            if trial < 2:
                draw_main_images()
                self.images['left_carpet_destination'].draw()
                self.msg_frame.draw()
                
                self.msg_text.text  = u"משמאל הנחתם את השטיח שמכושף "\
              u"לעוף להר {} …".format(
                  color_translations[common_transitions[isymbols[0]]['color'].lower()]
              )[::-1]  # היפוך הכיוון
                self.msg_text.pos = (50,405)
                self.msg_text.draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.images['right_carpet_destination'].draw()
                self.msg_frame.draw()
                self.msg_text.text = u"ומימין השטיח שמכושף "\
                     u"לעוף להר {} …".format(
                    color_translations[common_transitions[isymbols[1]]['color'].lower()]
                )[::-1]
                self.msg_text.pos = (500,405)
                self.msg_text.draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.images['tutorial_carpet_symbols'].draw()
                self.msg_frame.draw()
                self.msg_text.text = u"הסמלים שכתובים על השטיחים משמעותם ”ההר {}“ ו-”ההר {}“ בשפה המקומית.".format(
    color_translations[common_transitions[isymbols[0]]['color'].lower()],
    color_translations[common_transitions[isymbols[1]]['color'].lower()]
)[::-1]
                self.msg_text.pos = (625,405)
                self.msg_text.draw()
                self.win.flip()
                core.wait(5)
                

                draw_main_images()
                self.win.flip()
                core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            
            hebrew_text = u"בקרוב תוכלו לבחור שטיח ולעוף עליו על ידי לחיצה על המקש השמאלי או הימני."

            # Assign the reversed text to the TextStim object
            self.msg_text.text = hebrew_text[::-1]  # Reverse for proper RTL rendering

            # self.msg_text.text = 'You will soon be able to choose a carpet '\
            #     'and fly on it by pressing the left or right arrow key.'
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)
            

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            hebrew_text = u"כשהשטיחים מתחילים לזהור, יש לך 2 שניות ללחוץ על מקש, "\
              u"או שהם יעופו בלעדיך."
            self.msg_text.text = hebrew_text[::-1]  # Reverse for proper RTL rendering
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        elif trial < 5:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            hebrew_text = u"השטיחים שלך מוכנים ועומדים להתחיל לזהור. "\
              u"התכונן לעשות את הבחירה שלך."

            # Assign the reversed text to the TextStim object
            self.msg_text.text = hebrew_text[::-1]  # Reverse for proper RTL rendering

            self.msg_text.draw()
            self.win.flip()
            core.wait(3)

        # Glow carpets for response
        self.images['carpets_glow_tutorial'].draw()
        isymbols_image.draw()
        destination_image.draw()
        self.win.flip()
    def display_selected_carpet(self, trial, choice1, isymbols, common_transitions):
        isymbols_image = self.images['tibetan.{:02d}{:02d}'.format(*isymbols)]
        destination_image = self.images['carpets_to_{}_{}'.format(
            *[common_transitions[symbol]['color'] for symbol in isymbols]
        )]
        def draw_main_images():
            self.images['carpets_tutorial'].draw()
            isymbols_image.draw()
            destination_image.draw()
            self.images['tutorial_{}_carpet_selected'.format(choice1)].draw()
        if trial < 4:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            hebrew_text = u"בחרת בשטיח שב{}, שמכושף לעוף אל ההר {}. טיסה נעימה!".format(
            u'ימין' if choice1 == 's' else u'שמאל',
            color_translations[common_transitions[isymbols[int(choice1 == 's')]]['color'].lower()]
        )

            # Assign the reversed text for RTL display
            self.msg_text.text = hebrew_text[::-1]  # Reverse the text for proper RTL rendering

            self.msg_text.draw()
            self.win.flip()
            core.wait(5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(2)
    def display_transition(self, trial, final_state_color, common):
        transition_image = self.images['flight_{}-{}_{}{}'.format(
            final_state_color,
            self.mountain_sides[0],
            self.mountain_sides[1],
            '-wind' if not common else '',
        )]

        if common:
            hebrew_text = u"הטיסה שלך להר {} עברה היטב, בלי שום תקלות.".format(
             color_translations[final_state_color.lower()]
                )
            self.msg_text.text = hebrew_text[::-1]
        else:
            colors = TutorialConfig.final_state_colors
            hebrew_text = u"אוי לא! הרוחות ליד הר {} חזקות מדי. "\
              u"אתה מחליט לנחות עם השטיח שלך על הר {} במקום.".format(
            color_translations[colors[1 - colors.index(final_state_color)].lower()],
            color_translations[final_state_color.lower()]
            )

            self.msg_text.text = hebrew_text[::-1]  # PsychoPy requires reversed text for RTL languages

        if trial < 2:
            transition_image.draw()
            self.win.flip()
            core.wait(0.5)

            transition_image.draw()
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(3 if common else 6)

            transition_image.draw()
            self.win.flip()
            core.wait(0.5)
        else:
            transition_image.draw()
            self.win.flip()
            core.wait(2)
    def display_lamps(self, trial, final_state_color, fsymbols):
        fsymbols_image = self.images['tibetan.{:02}{:02}'.format(*fsymbols)]
        def draw_main_images():
            self.images['lamps_{}'.format(final_state_color)].draw()
            fsymbols_image.draw()

        if self.visits_to_mountains[final_state_color] < 1:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            
            hebrew_text = u"נחתת בבטחה על ההר {}.".format(
            color_translations[final_state_color.lower()])
            self.msg_text.text = hebrew_text[::-1]  # Reverse for RTL display in PsychoPy

            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(2)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            hebrew_text = u"הנה המנורות שבהן גרים הג'ינים של הר {}.".format(
            color_translations[final_state_color.lower()])

            # Reverse the text for proper RTL rendering in PsychoPy
            self.msg_text.text = hebrew_text[::-1]  # PsychoPy requires reversed text for RTL languages

            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            hebrew_text = u"המנורה משמאל היא ביתו של הג'יני "\
              u"ששמו מופיע למטה בשפה המקומית."

            # Reverse the text for proper RTL rendering in PsychoPy
            self.msg_text.text = hebrew_text[::-1]  # PsychoPy requires reversed text for RTL languages

            self.msg_frame.draw()
            self.msg_text.draw()
            self.images['left_lamp_symbol'].draw()
            self.win.flip()
            core.wait(4)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            hebrew_text = u"המנורה מימין היא ביתו של הג'יני "\
              u"ששמו מופיע למטה בשפה המקומית."

            # Reverse the text for proper RTL rendering in PsychoPy
            self.msg_text.text = hebrew_text[::-1]  # PsychoPy requires reversed text for RTL languages

            self.msg_frame.draw()
            self.msg_text.draw()
            self.images['right_lamp_symbol'].draw()
            self.win.flip()
            core.wait(3)
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            hebrew_text = u"בקרוב תוכלו לבחור מנורה ולשפשף אותה על ידי לחיצה "\
              u"על החץ השמאלי או הימני."

            # Reverse the text for proper RTL rendering in PsychoPy
            self.msg_text.text = hebrew_text[::-1]  # Reverse for RTL display

            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(4)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            hebrew_text = u"כשהמנורות מתחילות לזהור, יש לך 2 שניות ללחוץ על מקש, "\
              u"אחרת הג'ינים יחזרו לישון."

            # Reverse the text for proper RTL rendering in PsychoPy
            self.msg_text.text = hebrew_text[::-1]  # Reverse for RTL display
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)
            

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        elif trial < 5:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            hebrew_text = u"המנורות עומדות להתחיל לזהור, התכונן לבצע את הבחירה שלך."


            # Assign the reversed text to `self.msg_text.text` for RTL rendering
            self.msg_text.text = hebrew_text[::-1]
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

        self.images['lamps_{}_glow'.format(final_state_color)].draw()
        fsymbols_image.draw()
        self.win.flip()
        self.visits_to_mountains[final_state_color] += 1
    def display_selected_lamp(self, trial, final_state_color, fsymbols, choice2):
        fsymbols_image = self.images['tibetan.{:02}{:02}'.format(*fsymbols)]
        def draw_main_images():
            self.images['lamps_{}'.format(final_state_color)].draw()
            self.images['{}_lamp_selected'.format(choice2)].draw()
            fsymbols_image.draw()
        if trial < 5:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            hebrew_text = u"אתם מרימים את המנורה שב{} ומשפשפים אותה.".format(
            u'ימין' if choice2 == 's' else u'שמאל'  # Translate 's' and 'k' dynamically
            )

            # Assign the reversed text to `self.msg_text.text` for RTL rendering
            self.msg_text.text = hebrew_text[::-1]
            self.msg_text.draw()
            self.win.flip()
            core.wait(4)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(2)
    def display_reward(self, trial, final_state_color, chosen_symbol2):
        def draw_main_images():
            self.images['genie_coin'].draw()
            self.images['reward_{}'.format(final_state_color)].draw()
            self.images['tibetan.{:02}'.format(chosen_symbol2)].draw()

        if trial < 5:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)

            draw_main_images()
            self.msg_frame.draw()
            hebrew_text = u"הג'יני יצא מהמנורה שלו, הקשיב לשיר, ונתן לך מטבע זהב!"

            # Assign the reversed text to `self.msg_text.text` for RTL rendering
            self.msg_text.text = hebrew_text[::-1]
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            if trial < 2:
                draw_main_images()
                self.msg_frame.draw()
                hebrew_text = u"זכרו את שמו של הג'יני הזה למקרה שתרצו לבחור שוב את המנורה שלו בעתיד."

                # Assign the reversed text to `self.msg_text.text` for RTL rendering
                self.msg_text.text = hebrew_text[::-1]
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.msg_frame.draw()
                hebrew_text = u"הצבע של המנורה שלו מזכיר לך שהוא גר על ההר {}.".format(
                color_translations[final_state_color.lower()]  # Translate the color dynamically
                )

                # Assign the reversed text to `self.msg_text.text` for RTL rendering
                self.msg_text.text = hebrew_text[::-1]
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(3)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)
    def display_no_reward(self, trial, final_state_color, chosen_symbol2):
        def draw_main_images():
            self.images['genie_zero'].draw()
            self.images['reward_{}'.format(final_state_color)].draw()
            self.images['tibetan.{:02}'.format(chosen_symbol2)].draw()

        if trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)

            draw_main_images()
            self.msg_frame.draw()
            hebrew_text = u"הג'יני נשאר בתוך המנורה שלו, ולא קיבלת מטבע זהב."

            # Assign the reversed text to `self.msg_text.text` for RTL rendering
            self.msg_text.text = hebrew_text[::-1]
            self.msg_text.draw()
            self.win.flip()
            core.wait(3)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            if trial < 2:
                draw_main_images()
                self.msg_frame.draw()
                hebrew_text = u"זכור את שמו של הג'יני הזה למקרה שתרצה לבחור שוב את המנורה שלו בעתיד."

                # Assign the reversed text to `self.msg_text.text` for RTL rendering
                self.msg_text.text = hebrew_text[::-1]
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.msg_frame.draw()
                hebrew_text = u"הצבע של המנורה שלו מזכיר לך שהוא גר על ההר {}.".format(
                color_translations[final_state_color.lower()]  # Translate the color dynamically
                )

                # Assign the reversed text to `self.msg_text.text` for RTL rendering
                self.msg_text.text = hebrew_text[::-1]
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(3)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)
    def display_end_of_trial(self):
        pass
    def display_slow1(self):
        self.images['slow1'].draw()
        self.win.flip()
        core.wait(4)
    def display_slow2(self, final_state_color):
        self.images['lamps_{}'.format(final_state_color)].draw()
        self.images['slow2'].draw()
        self.win.flip()
        core.wait(4)
    def display_break(self):
        self.images['break'].draw()
        self.win.flip()

if __name__ == '__main__':
    main()
